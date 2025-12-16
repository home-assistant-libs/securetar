"""Tarfile fileobject handler for encrypted files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Hashable
import copy
import enum
import hashlib
import logging
import os
import struct
import tarfile
import time
from pathlib import Path, PurePath
from types import NoneType
from typing import TYPE_CHECKING, Any, IO, BinaryIO, Literal

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

# Sizes for AES, used in v1 and v2
AES_BLOCK_SIZE = 16
AES_BLOCK_SIZE_BITS = 128
AES_IV_SIZE = AES_BLOCK_SIZE

DEFAULT_BUFSIZE = 10240

# The initial header consists of 16 bytes of magic, version, reserved.
# The 16 bytes size is chosen to align with v1 which has no magic, just 16 byte AES IV.
SECURETAR_MAGIC = b"SecureTar"
SECURETAR_MAGIC_RESERVED = b"\x00" * 6

# SecureTar v1 header consists of:
# 0 bytes file ID (no magic)
# 16 bytes AES IV
SECURETAR_V1_FILE_ID_SIZE = 0
SECURETAR_V1_CIPHER_INIT_SIZE = AES_IV_SIZE
SECURETAR_V1_HEADER_SIZE = SECURETAR_V1_FILE_ID_SIZE + SECURETAR_V1_CIPHER_INIT_SIZE

# SecureTar v2 header consists of:
# 16 bytes file ID: 9 bytes magic + 1 byte version + 6 bytes reserved
# 16 bytes file metadata: 8 bytes plaintext size + 8 bytes reserved
# 16 bytes AES IV
SECURETAR_V2_FILE_ID_FORMAT = "!9sB6s"
SECURETAR_V2_FILE_METADATA_FORMAT = "!Q8x"
SECURETAR_V2_CIPHER_INIT_SIZE = AES_IV_SIZE
SECURETAR_V2_HEADER_SIZE = (
    struct.calcsize(SECURETAR_V2_FILE_ID_FORMAT)
    + struct.calcsize(SECURETAR_V2_FILE_METADATA_FORMAT)
    + SECURETAR_V2_CIPHER_INIT_SIZE
)


GZIP_MAGIC_BYTES = b"\x1f\x8b\x08"
TAR_MAGIC_BYTES = b"ustar"
TAR_MAGIC_OFFSET = 257

TAR_BLOCK_SIZE = 512

MOD_EXCLUSIVE = "x"
MOD_READ = "r"
MOD_WRITE = "w"


class CipherMode(enum.Enum):
    """Cipher mode."""

    ENCRYPT = 1
    DECRYPT = 2


class SecureTarHeader:
    """SecureTar header.

    Reads and produces the SecureTar header. Also accepts the magic-less
    format used in earlier releases of SecureTar.
    """

    def __init__(
        self, cipher_initialization: bytes, plaintext_size: int | None, version: int
    ) -> None:
        """Initialize SecureTar header."""
        self.cipher_initialization = cipher_initialization
        self.plaintext_size = plaintext_size
        if version not in (1, 2):
            raise ValueError(f"Unsupported SecureTar version: {version}")
        self.version = version

        if version == 1:
            self.size = SECURETAR_V1_HEADER_SIZE
        else:
            self.size = SECURETAR_V2_HEADER_SIZE

    @classmethod
    def from_bytes(cls, f: IO[bytes]) -> SecureTarHeader:
        """Create from bytes."""
        # Read magic, version (1 byte), reserved
        header = f.read(struct.calcsize(SECURETAR_V2_FILE_ID_FORMAT))
        plaintext_size: int | None = None
        magic, version, reserved = struct.unpack(SECURETAR_V2_FILE_ID_FORMAT, header)
        if (
            magic != SECURETAR_MAGIC
            or version not in (2, 3)
            or reserved != SECURETAR_MAGIC_RESERVED
        ):
            # Assume legacy format without magic
            cipher_initialization = header
            version = 1
        else:
            file_metadata = f.read(struct.calcsize(SECURETAR_V2_FILE_METADATA_FORMAT))
            # Valid SecureTar v2+ header, read rest of header: plaintext size + reserved
            (plaintext_size,) = struct.unpack(
                SECURETAR_V2_FILE_METADATA_FORMAT, file_metadata
            )
            cipher_initialization = f.read(SECURETAR_V2_CIPHER_INIT_SIZE)

        return cls(cipher_initialization, plaintext_size, version)

    def to_bytes(self) -> bytes:
        """Return header bytes."""
        if self.plaintext_size is None:
            raise ValueError("Plaintext size is required")
        # Check version.
        # SecureTar versions writing v1 had bugs related to how the padding was
        # handled, and we don't support such archives anymore.
        if self.version != 2:
            raise ValueError(f"Unsupported SecureTar version: {self.version}")

        return (
            struct.pack(
                SECURETAR_V2_FILE_ID_FORMAT,
                SECURETAR_MAGIC,
                self.version,
                SECURETAR_MAGIC_RESERVED,
            )
            + struct.pack(SECURETAR_V2_FILE_METADATA_FORMAT, self.plaintext_size)
            + self.cipher_initialization
        )


class SecureTarError(Exception):
    """SecureTar error."""


class AddFileError(SecureTarError):
    """Raised when a file could not be added to an archive."""

    def __init__(self, path: Path, *args: Any) -> None:
        """Initialize."""
        self.path = path
        super().__init__(*args)


class SecureTarReadError(SecureTarError):
    """SecureTar read error."""


def _is_valid_tar_header(data: bytes) -> bool:
    """Check if data looks like a valid tar or gzip header."""
    is_gzip = data[: len(GZIP_MAGIC_BYTES)] == GZIP_MAGIC_BYTES
    is_tar = (
        data[TAR_MAGIC_OFFSET : TAR_MAGIC_OFFSET + len(TAR_MAGIC_BYTES)]
        == TAR_MAGIC_BYTES
    )
    return is_gzip or is_tar


class CipherStream(ABC):
    """Abstract base for cipher stream operations."""

    @abstractmethod
    def close(self) -> None:
        """Close the stream."""


class CipherReader(CipherStream):
    """Abstract base for cipher readers with buffered read."""

    def __init__(self, source: IO[bytes], ciphertext_size: int | None = None) -> None:
        """Initialize cipher reader."""
        self._buffer = b""
        self._ciphertext_size = ciphertext_size
        self._done = False
        self._source = source

    def read(self, size: int = 0) -> bytes:
        """Read data from buffer, filling as needed."""
        if not self._done:
            self._fill_buffer(size)

        data = self._buffer[:size]
        self._buffer = self._buffer[size:]
        return data

    @abstractmethod
    def _fill_buffer(self, size: int) -> None:
        """Fill buffer with at least size bytes if possible."""


class DecryptReader(CipherReader):
    """Abstract base for decryption readers."""

    def __init__(
        self,
        source: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
        ciphertext_size: int | None = None,
    ) -> None:
        """Initialize decryption reader."""
        super().__init__(source, ciphertext_size)


class EncryptWriter(CipherStream):
    """Abstract base for encryption writers."""

    def __init__(
        self,
        dest: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
    ) -> None:
        """Initialize encryption writer."""
        self._dest = dest
        self._plaintext_size = 0

    def write(self, data: bytes) -> None:
        """Write plaintext data to be encrypted."""
        self._plaintext_size += len(data)
        self._write(data)

    @abstractmethod
    def _write(self, data: bytes) -> None:
        """Write plaintext data to be encrypted."""

    @property
    def plaintext_size(self) -> int:
        """Return the total plaintext bytes written."""
        return self._plaintext_size


class EncryptReader(CipherReader):
    """Abstract base for encryption readers (for streaming encryption)."""

    _ciphertext_size: int

    def __init__(
        self,
        source: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
        plaintext_size: int,
    ) -> None:
        """Initialize encryption reader."""
        super().__init__(source, 0)
        self._plaintext_size = plaintext_size

    @property
    def ciphertext_size(self) -> int:
        """Return the total ciphertext size."""
        return self._ciphertext_size


class CipherStreamFactory(ABC):
    """Abstract factory for creating cipher streams."""

    create_decrypt_reader: type[DecryptReader]
    """Create a decryption reader."""

    create_encrypt_writer: type[EncryptWriter]
    """Create an encryption writer."""

    create_encrypt_reader: type[EncryptReader]
    """Create an encryption reader (for streaming encryption with known size)."""


class _AesCbcDecryptReader(DecryptReader):
    """AES-CBC decryption reader.

    AES-CBC is used in SecureTar v1 and v2.
    """

    def __init__(
        self,
        source: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
        ciphertext_size: int | None = None,
    ) -> None:
        """Initialize."""
        super().__init__(source, key_material, ciphertext_size)
        self._pos = 0
        self._validated = False

        aes = Cipher(
            algorithms.AES(key_material.key),
            modes.CBC(key_material.iv),
            backend=default_backend(),
        )
        self._cipher = aes.decryptor()

    def _validate(self) -> None:
        """Validate decrypted data looks like tar/gzip."""
        if self._validated:
            return

        # Read first block to validate
        encrypted = self._source.read(TAR_BLOCK_SIZE)
        self._pos = len(encrypted)
        self._buffer = self._cipher.update(encrypted)

        if not _is_valid_tar_header(self._buffer):
            raise SecureTarReadError("The inner tar is not gzip or tar, wrong key?")

        self._validated = True

    def _fill_buffer(self, size: int) -> None:
        """Fill buffer with decrypted data."""
        if not self._validated:
            self._validate()

        # Determine how much more to read
        to_read = max(size, DEFAULT_BUFSIZE)
        if self._ciphertext_size is not None:
            # Bounded: read up to remaining ciphertext
            remaining = self._ciphertext_size - self._pos
            to_read = min(to_read, remaining)

        while len(self._buffer) < size + AES_BLOCK_SIZE and not self._done:
            encrypted_data = self._source.read(to_read)
            if not encrypted_data:
                # EOF - strip padding
                self._strip_padding()
                self._done = True
                break

            self._pos += len(encrypted_data)
            self._buffer += self._cipher.update(encrypted_data)

            # Check if we've read all ciphertext (bounded case)
            if self._ciphertext_size is not None and self._pos >= self._ciphertext_size:
                self._strip_padding()
                self._done = True
                break

    def _strip_padding(self) -> None:
        """Strip PKCS7 padding from buffer."""
        if self._buffer:
            padding_len = self._buffer[-1]
            self._buffer = self._buffer[:-padding_len]

    def close(self) -> None:
        """Close the decrypt reader."""
        self._cipher = None


class _AesCbcEncryptWriter(EncryptWriter):
    """AES-CBC encryption writer.

    AES-CBC is used in SecureTar v1 and v2.
    """

    def __init__(
        self,
        dest: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
    ) -> None:
        """Initialize."""
        super().__init__(dest, key_material)

        aes = Cipher(
            algorithms.AES(key_material.key),
            modes.CBC(key_material.iv),
            backend=default_backend(),
        )
        self._cipher = aes.encryptor()
        self._padder = padding.PKCS7(AES_BLOCK_SIZE_BITS).padder()

    def _write(self, data: bytes) -> None:
        """Write plaintext data to be encrypted."""
        padded = self._padder.update(data)
        if padded:
            self._dest.write(self._cipher.update(padded))

    def close(self) -> None:
        """Close the encrypt writer."""
        if self._padder and self._cipher:
            final_padding = self._padder.finalize()
            self._dest.write(self._cipher.update(final_padding))
        self._padder = None
        self._cipher = None


class _AesCbcEncryptReader(EncryptReader):
    """AES-CBC encryption reader (for streaming encryption).

    AES-CBC is used in SecureTar v1 and v2.
    """

    def __init__(
        self,
        source: IO[bytes],
        key_material: SecureTarDerivedKeyMaterial,
        plaintext_size: int,
    ) -> None:
        """Initialize."""
        super().__init__(source, key_material, plaintext_size)
        self._ciphertext_size = (
            plaintext_size + AES_BLOCK_SIZE - (plaintext_size % AES_BLOCK_SIZE)
        )
        self._pos = 0

        self._cipher = Cipher(
            algorithms.AES(key_material.key),
            modes.CBC(key_material.iv),
            backend=default_backend(),
        ).encryptor()
        self._padder = padding.PKCS7(AES_BLOCK_SIZE_BITS).padder()

    def _fill_buffer(self, size: int) -> None:
        """Fill buffer with encrypted data."""
        while len(self._buffer) < size and not self._done:
            remaining = self._plaintext_size - self._pos
            to_read = min(max(size, DEFAULT_BUFSIZE), remaining) if remaining > 0 else 0

            if to_read > 0:
                plaintext = self._source.read(to_read)
                self._pos += len(plaintext)
                padded = self._padder.update(plaintext)
                if padded:
                    self._buffer += self._cipher.update(padded)

            if self._pos >= self._plaintext_size:
                # Finalize
                final_padding = self._padder.finalize()
                self._buffer += self._cipher.update(final_padding)
                self._done = True

    def close(self) -> None:
        """Close the encrypt reader."""
        self._cipher = None
        self._padder = None


class AesCbcStreamFactory(CipherStreamFactory):
    """Factory for AES-CBC streams (v1/v2)."""

    create_decrypt_reader = _AesCbcDecryptReader
    create_encrypt_writer = _AesCbcEncryptWriter
    create_encrypt_reader = _AesCbcEncryptReader


class SecureTarDecryptStream:
    """Decrypts an encrypted tar read from a stream."""

    def __init__(
        self,
        source: IO[bytes],
        *,
        ciphertext_size: int | None = None,
        root_key_context: SecureTarRootKeyContext,
    ) -> None:
        """Initialize."""
        self._source = source
        self._root_key_context = root_key_context
        self._ciphertext_size = ciphertext_size
        self._stream: DecryptReader | None = None
        self._header: SecureTarHeader | None = None

    def __enter__(self) -> DecryptReader:
        self._header = SecureTarHeader.from_bytes(self._source)
        key_material = self._root_key_context.restore_key_material(self._header)

        ciphertext_size = None
        if self._ciphertext_size is not None:
            ciphertext_size = self._ciphertext_size - self._header.size
        self._stream = self._root_key_context.stream_factory.create_decrypt_reader(
            self._source,
            key_material,
            ciphertext_size,
        )
        return self._stream

    def __exit__(self, exc_type, exc_value, tb) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None

    def validate(self) -> bool:
        """Validate the password by checking if decrypted data looks like a tar.

        Note: This consumes the stream. Create a new instance to read data.

        Returns:
            True if password is valid, False otherwise.
        """
        try:
            with self as stream:
                stream.read(1)
                return True
        except SecureTarReadError:
            return False


class _FramedEncryptReader:
    """Wraps an encrypt reader with a SecureTar header."""

    def __init__(self, inner: EncryptReader, header: SecureTarHeader) -> None:
        """Initialize."""
        self._inner = inner
        self._header = header
        self._header_bytes: bytes | None = header.to_bytes()
        self._ciphertext_size = header.size + inner.ciphertext_size

    @property
    def ciphertext_size(self) -> int:
        """Return the size of the ciphertext data."""
        return self._ciphertext_size

    def read(self, size: int = 0) -> bytes:
        """Read data from buffer, filling as needed.

        Will first return header bytes, then data from inner stream.
        """
        data = b""

        if self._header_bytes:
            data = self._header_bytes[:size]
            self._header_bytes = (
                self._header_bytes[size:] if len(self._header_bytes) > size else None
            )
            size -= len(data)
            if size == 0:
                return data

        data += self._inner.read(size)
        return data

    def close(self) -> None:
        """Close the stream."""
        self._inner.close()


class SecureTarEncryptStream:
    """Encrypt a plaintext tar read from a stream."""

    def __init__(
        self,
        source: IO[bytes],
        *,
        derived_key_id: Hashable | None,
        plaintext_size: int,
        root_key_context: SecureTarRootKeyContext,
    ) -> None:
        """Initialize."""
        self._source = source
        self._derived_key_id = derived_key_id
        self._plaintext_size = plaintext_size
        self._root_key_context = root_key_context
        self._stream: _FramedEncryptReader | None = None

    def __enter__(self) -> _FramedEncryptReader:
        key_material = self._root_key_context.derive_key_material(
            self._derived_key_id, 2
        )

        factory = self._root_key_context.stream_factory
        inner_stream = factory.create_encrypt_reader(
            self._source,
            key_material,
            self._plaintext_size,
        )

        header = SecureTarHeader(
            key_material.cipher_initialization,
            self._plaintext_size,
            2,
        )

        self._stream = _FramedEncryptReader(inner_stream, header)
        return self._stream

    def __exit__(self, exc_type, exc_value, tb) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None


class SecureTarFile:
    """Handle tar files, optionally wrapped in an encryption layer."""

    def __init__(
        self,
        name: Path | None = None,
        mode: Literal["r", "w", "x"] = "r",
        *,
        bufsize: int = DEFAULT_BUFSIZE,
        derived_key_id: Hashable | None = None,
        fileobj: IO[bytes] | None = None,
        gzip: bool = True,
        password: str | None = None,
        root_key_context: SecureTarRootKeyContext | None = None,
    ) -> None:
        """Initialize encryption handler.

        Args:
            name: Path to the tar file
            mode: File mode ('r' for read, 'w' for write, 'x' for exclusive create)
            bufsize: Buffer size for I/O operations
            derived_key_id: Optional derived key ID for deriving key material. Mutually
            exclusive with password.
            fileobj: File object to use instead of opening a file
            gzip: Whether to use gzip compression
            password: Password to derive encryption key from. Mutually exclusive with
            root_key_context and derived_key_id.
            root_key_context: Root key context to use for deriving key material. Mutually
            exclusive with password.
        """

        if (
            derived_key_id is not None or root_key_context is not None
        ) and password is not None:
            raise ValueError("Cannot specify both 'derived_key_id' and 'password'")
        if derived_key_id is not None and root_key_context is None:
            raise ValueError(
                "Cannot specify 'derived_key_id' without 'root_key_context'"
            )

        if name is None and fileobj is None:
            raise ValueError("Either filename or fileobj must be provided")

        if mode not in (MOD_EXCLUSIVE, MOD_READ, MOD_WRITE):
            raise ValueError(
                f"Mode must be '{MOD_EXCLUSIVE}', '{MOD_READ}', or '{MOD_WRITE}'"
            )

        self._file: IO[bytes] | None = None
        self._mode: str = mode
        self._name: Path | None = name
        self._bufsize: int = bufsize
        self._extra_tar_args: dict[str, Any] = {}
        self._fileobj = fileobj
        self._tar: tarfile.TarFile | None = None
        self._derived_key_id = derived_key_id
        self._cipher_stream: CipherStream | None = None
        self._header: SecureTarHeader | None = None

        # Determine if encrypted
        self._encrypted = password is not None or root_key_context is not None

        # Determine the mode for tarfile.open(), streaming (|) for encrypted because
        # we can't seek in the ciphertext, normal (:) for plain
        if self._encrypted:
            self._tar_mode = f"{mode}|"
        else:
            self._tar_mode = f"{mode}:"
            if gzip:
                self._extra_tar_args["compresslevel"] = 6
        if gzip:
            self._tar_mode += "gz"

        # Set up key context
        self._root_key_context: SecureTarRootKeyContext | None = None
        if self._encrypted:
            if password:
                self._root_key_context = SecureTarRootKeyContext(password)
            else:
                self._root_key_context = root_key_context

    def __enter__(self) -> tarfile.TarFile:
        """Start context manager tarfile."""
        return self.open()

    def open(self) -> tarfile.TarFile:
        """Open SecureTar file.

        Returns tarfile object, data written to is encrypted if key is provided.
        """
        if not self._encrypted:
            # Plain tar, no encryption
            self._tar = tarfile.open(
                name=str(self._name),
                mode=self._tar_mode,
                dereference=False,
                bufsize=self._bufsize,
                fileobj=self._fileobj,
                **self._extra_tar_args,
            )
            return self._tar

        # When encrypted, we have root key context
        if TYPE_CHECKING:
            assert self._root_key_context is not None

        # Open underlying file or stream
        self._file = self._open_file()

        # Set up cipher layer
        if self._mode == MOD_READ:
            self._header = SecureTarHeader.from_bytes(self._file)
            key_material = self._root_key_context.restore_key_material(self._header)
            factory = self._root_key_context.stream_factory
            self._cipher_stream = factory.create_decrypt_reader(
                self._file, key_material
            )
        else:
            key_material = self._root_key_context.derive_key_material(
                self._derived_key_id, 2
            )
            self._header = SecureTarHeader(
                key_material.cipher_initialization,
                0,
                2,
            )
            self._file.write(self._header.to_bytes())
            factory = self._root_key_context.stream_factory
            self._cipher_stream = factory.create_encrypt_writer(
                self._file, key_material
            )

        # Open tar with cipher as fileobj
        self._tar = tarfile.open(
            fileobj=self._cipher_stream,
            mode=self._tar_mode,
            dereference=False,
            bufsize=self._bufsize,
        )
        return self._tar

    def _open_file(self) -> IO[bytes] | BinaryIO:
        if self._fileobj:
            # If we have a fileobj, we don't need to open a file
            return self._fileobj
        else:
            # We check in constructor that name and fileobj are not both None
            if TYPE_CHECKING:
                assert self._name is not None
            read_mode = self._mode.startswith("r")
            if read_mode:
                file_mode: int = os.O_RDONLY
            else:
                file_mode = os.O_WRONLY | os.O_CREAT

            fd = os.open(self._name, file_mode, 0o666)
            return os.fdopen(fd, "rb" if read_mode else "wb")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close file."""
        if self._tar:
            self._tar.close()
            self._tar = None
        if self._cipher_stream:
            self._cipher_stream.close()
            self._cipher_stream = None
        if self._file:
            if not self._fileobj:
                self._file.close()
            self._file = None

    def validate_password(self) -> bool:
        """Validate the password by checking if decrypted data looks like a tar.

        Note: If using fileobj instead of a file path, this consumes the stream
        and a new SecureTarFile instance must be created to read data.

        Returns:
            True if password is valid, False otherwise.

        Raises:
            SecureTarError: If file is not encrypted, not in read mode, or already open.
        """
        if not self._encrypted:
            raise SecureTarError("File is not encrypted")

        if self._mode != MOD_READ:
            raise SecureTarError("Can only validate password in read mode")

        if self._tar is not None:
            raise SecureTarError("File is already open")

        if TYPE_CHECKING:
            assert self._root_key_context is not None

        file = self._open_file()
        try:
            return SecureTarDecryptStream(
                file,
                root_key_context=self._root_key_context,
            ).validate()
        finally:
            if not self._fileobj:
                file.close()

    @property
    def path(self) -> Path:
        """Return path object of tarfile."""
        return self._name

    @property
    def size(self) -> float:
        """Return backup size."""
        if not self._name or not self._name.is_file():
            return 0
        return round(self._name.stat().st_size / 1_048_576, 2)  # calc mbyte


class InnerSecureTarFile(SecureTarFile):
    """Handle encrypted files for tarfile library inside another tarfile."""

    _header_length: int
    _header_position: int
    _tar_info: tarfile.TarInfo

    def __init__(
        self,
        outer_tar: tarfile.TarFile,
        name: Path,
        mode: Literal["r", "w", "x"],
        *,
        bufsize: int,
        derived_key_id: Hashable | None,
        gzip: bool,
        root_key_context: SecureTarRootKeyContext | None,
    ) -> None:
        """Initialize inner handler."""
        super().__init__(
            name=name,
            mode=mode,
            gzip=gzip,
            bufsize=bufsize,
            derived_key_id=derived_key_id,
            fileobj=outer_tar.fileobj,
            root_key_context=root_key_context,
        )
        self.outer_tar = outer_tar

    def __enter__(self) -> tarfile.TarFile:
        """Start context manager tarfile."""
        self._tar_info = tarfile.TarInfo(name=str(self._name))
        if self.outer_tar.format == tarfile.PAX_FORMAT:
            # Ensure we always set mtime as a float to force
            # a PAX header to be written.
            #
            # This is necessary to
            # handle large files as TarInfo.tobuf will try to
            # use a shorter ustar header if we do not have at
            # least one float in the tarinfo.
            # https://github.com/python/cpython/blob/53b84e772cac6e4a55cebf908d6bb9c48fe254dc/Lib/tarfile.py#L1066
            self._tar_info.mtime = time.time()
        else:
            self._tar_info.mtime = int(time.time())

        fileobj = self.outer_tar.fileobj
        if fileobj is None:
            raise ValueError("Outer tar file has no fileobj")

        self._header_position = fileobj.tell()

        # Write an empty header for the inner tar file in the outer tar file
        # We'll seek back to this position later to update the header with the correct size
        outer_tar = self.outer_tar
        tar_info_header = self._tar_info.tobuf(
            outer_tar.format, outer_tar.encoding, outer_tar.errors
        )
        self._header_length = len(tar_info_header)
        fileobj.write(tar_info_header)

        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close file."""
        # This class is only used for writing inner tar files, so we know
        # that if _cipher_stream is set, it is an EncryptWriter.
        if TYPE_CHECKING:
            assert isinstance(self._cipher_stream, (EncryptWriter, NoneType))
        # Capture encrypt_writer before super().__exit__ sets it to None
        encrypt_writer: EncryptWriter | None = self._cipher_stream  # type: ignore[assignment]
        super().__exit__(exc_type, exc_value, traceback)
        self._finalize_tar_entry(encrypt_writer)

    def _finalize_tar_entry(self, encrypt_writer: EncryptWriter | None) -> None:
        """Update tar header and securetar header with final sizes."""
        outer_tar = self.outer_tar
        fileobj = self.outer_tar.fileobj
        if fileobj is None:
            raise ValueError("Outer tar file has no fileobj")

        end_position = fileobj.tell()

        size_of_inner_tar = end_position - self._header_position - self._header_length

        # Pad to tar block boundary
        blocks, remainder = divmod(size_of_inner_tar, tarfile.BLOCKSIZE)
        padding_size = 0
        if remainder > 0:
            padding_size = tarfile.BLOCKSIZE - remainder
            fileobj.write(tarfile.NUL * padding_size)
        outer_tar.offset += size_of_inner_tar + padding_size

        # Update securetar header with plaintext size if encrypted
        if self._encrypted and self._header and encrypt_writer:
            self._header.plaintext_size = encrypt_writer.plaintext_size
            fileobj.seek(self._header_position + self._header_length)
            fileobj.write(self._header.to_bytes())

        # Now that we know the size of the inner tar, we seek back
        # to where we started and re-write the header with the correct size
        self._tar_info.size = size_of_inner_tar
        fileobj.seek(self._header_position)

        # We can't call tar.addfile here because it doesn't allow a non-zero
        # size to be set without passing a fileobj. Instead we manually write
        # the header. https://github.com/python/cpython/pull/117988
        buf = self._tar_info.tobuf(
            outer_tar.format, outer_tar.encoding, outer_tar.errors
        )
        fileobj.write(buf)
        outer_tar.offset += len(buf)
        outer_tar.members.append(self._tar_info)

        # Finally return to the end of the outer tar file
        fileobj.seek(end_position + padding_size)


class SecureTarArchive:
    """Manage a plain tar archive containing encrypted inner tar files."""

    def __init__(
        self,
        name: Path | None = None,
        mode: Literal["r", "w"] = "r",
        *,
        bufsize: int = DEFAULT_BUFSIZE,
        fileobj: IO[bytes] | None = None,
        password: str | None = None,
        root_key_context: SecureTarRootKeyContext | None = None,
        streaming: bool = False,
    ) -> None:
        """Initialize archive handler.

        Args:
            name: Path to the tar file
            mode: File mode ('r' for read, 'w' for write, 'x' for exclusive create)
            bufsize: Buffer size for I/O operations
            fileobj: File object to use instead of opening a file
            password: Password for encryption/decryption of inner tar files. Mutually
            exclusive with root_key_context.
            root_key_context: Root key context to use for deriving key material. Mutually
            exclusive with password.
            streaming: Whether to use streaming mode for tarfile (no seeking)
        """
        if root_key_context is not None and password is not None:
            raise ValueError("Cannot specify both 'root_key_context' and 'password'")
        if name is None and fileobj is None:
            raise ValueError("Either name or fileobj must be provided")

        if mode not in (MOD_EXCLUSIVE, MOD_READ, MOD_WRITE):
            raise ValueError(
                f"Mode must be '{MOD_EXCLUSIVE}', '{MOD_READ}', or '{MOD_WRITE}'"
            )

        self._name = name
        self._mode = mode
        self._bufsize = bufsize
        self._fileobj = fileobj
        self._streaming = streaming
        self._tar: tarfile.TarFile | None = None

        # Set up key context
        self._root_key_context = root_key_context
        if password:
            self._root_key_context = SecureTarRootKeyContext(password)

    def __enter__(self) -> SecureTarArchive:
        """Open the archive."""
        return self.open()

    def open(self) -> SecureTarArchive:
        """Open the archive."""
        mode = f"{self._mode}{'|' if self._streaming else ''}"
        self._tar = tarfile.open(
            name=str(self._name) if self._name else None,
            mode=mode,
            fileobj=self._fileobj,
            bufsize=self._bufsize,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the archive."""
        self.close()

    def close(self) -> None:
        """Close the archive."""
        if self._tar:
            self._tar.close()
            self._tar = None

    @property
    def tar(self) -> tarfile.TarFile:
        """Return the underlying tar file."""
        if not self._tar:
            raise SecureTarError("Archive not open")
        return self._tar

    def create_tar(
        self,
        name: str,
        *,
        derived_key_id: Hashable | None = None,
        gzip: bool = True,
    ) -> InnerSecureTarFile:
        """Create a new tar file within the archive.

        Returns a context manager that yields a TarFile for adding files.
        The tar file will be encrypted if a password was provided to the archive.

        Args:
            name: Name of the inner tar file in the archive
            derived_key_id: Optional derived key ID for deriving key material.
            gzip: Whether to use gzip compression
        """
        if not self._tar:
            raise SecureTarError("Archive not open")
        if self._mode == MOD_READ:
            raise SecureTarError("Archive not open for writing")
        if self._streaming:
            raise SecureTarError("create_tar not supported in streaming mode")
        if derived_key_id is not None and self._root_key_context is None:
            raise ValueError(
                "Cannot specify 'derived_key_id' when encryption is disabled"
            )

        return InnerSecureTarFile(
            self._tar,
            bufsize=self._bufsize,
            gzip=gzip,
            mode="w",
            name=Path(name),
            derived_key_id=derived_key_id,
            root_key_context=self._root_key_context,
        )

    def extract_tar(
        self,
        member: tarfile.TarInfo,
    ) -> SecureTarDecryptStream:
        """Extract and decrypt a tar file from the archive.

        Returns a context manager that yields a readable stream of decrypted bytes.

        Args:
            member: TarInfo of the tar file to extract
        """
        if not self._tar:
            raise SecureTarError("Archive not open")

        if self._mode != MOD_READ:
            raise SecureTarError("Archive not open for reading")

        if not self._root_key_context:
            raise SecureTarError("No password provided")

        fileobj = self._tar.extractfile(member)
        if fileobj is None:
            raise SecureTarError(f"Cannot extract {member.name}")

        return SecureTarDecryptStream(
            fileobj,
            ciphertext_size=member.size,
            root_key_context=self._root_key_context,
        )

    def import_tar(
        self,
        source: IO[bytes],
        member: tarfile.TarInfo,
        *,
        derived_key_id: Hashable | None = None,
    ) -> None:
        """Import an existing tar into the archive, encrypting it.

        The member.size must be set to the size of the source stream.

        Args:
            source: Source tar stream to encrypt and add
            member: TarInfo for the tar file (size must be set)
            derived_key_id: Optional derived key ID for deriving key material.
        """
        if not self._tar:
            raise SecureTarError("Archive not open")

        if self._mode == MOD_READ:
            raise SecureTarError("Archive not open for writing")

        if not self._root_key_context:
            raise SecureTarError("No password provided")

        with SecureTarEncryptStream(
            source,
            derived_key_id=derived_key_id,
            plaintext_size=member.size,
            root_key_context=self._root_key_context,
        ) as encrypted:
            encrypted_tar_info = copy.deepcopy(member)
            encrypted_tar_info.size = encrypted.ciphertext_size
            self._tar.addfile(encrypted_tar_info, encrypted)

    def validate_password(self, member: tarfile.TarInfo) -> bool:
        """Validate the password against an encrypted inner tar.

        Note: This consumes the stream. Create a new instance to read data.

        Args:
            member: TarInfo of an encrypted tar file to validate against
        """
        if not self._tar:
            raise SecureTarError("Archive not open")

        if self._mode != MOD_READ:
            raise SecureTarError("Archive not open for reading")

        if not self._root_key_context:
            raise SecureTarError("No password provided")

        return self.extract_tar(member).validate()


class KeyDerivationStrategy(ABC):
    """Abstract base for version-specific key derivation."""

    stream_factory: type[CipherStreamFactory]
    """The cipher stream factory for this version."""

    @abstractmethod
    def get_key_material(
        self, cipher_initialization: bytes | None = None
    ) -> SecureTarDerivedKeyMaterial:
        """Get key material, deriving new or restoring from cipher_initialization."""


class KeyDerivationV2(KeyDerivationStrategy):
    """Key derivation for SecureTar v1 and v2."""

    stream_factory = AesCbcStreamFactory

    def __init__(self, password: str) -> None:
        """Initialize."""
        self._root_key = self._password_to_key(password)

    @staticmethod
    def _password_to_key(password: str) -> bytes:
        """Generate AES key from password using 100 rounds of SHA256."""
        key: bytes = password.encode()
        for _ in range(100):
            key = hashlib.sha256(key).digest()
        return key[:16]

    def get_key_material(
        self, cipher_initialization: bytes | None = None
    ) -> SecureTarDerivedKeyMaterialV2:
        """Get key material, deriving new or restoring from cipher_initialization."""
        salt = (
            cipher_initialization
            if cipher_initialization is not None
            else os.urandom(AES_IV_SIZE)
        )
        return SecureTarDerivedKeyMaterialV2(root_key=self._root_key, salt=salt)


class SecureTarDerivedKeyMaterial(ABC):
    """Abstract base for derived key material."""

    key: bytes
    """The derived encryption key."""

    iv: bytes
    """The initialization vector."""

    @property
    @abstractmethod
    def cipher_initialization(self) -> bytes:
        """Return cipher initialization bytes for the header."""


class SecureTarDerivedKeyMaterialV2(SecureTarDerivedKeyMaterial):
    """Key material for SecureTar v1 and v2."""

    def __init__(self, root_key: bytes, salt: bytes) -> None:
        """Initialize."""
        self.key = root_key
        self._salt = salt
        self.iv = self._generate_iv(root_key, salt)

    @staticmethod
    def _generate_iv(key: bytes, salt: bytes) -> bytes:
        """Generate an IV from key and salt."""
        temp_iv = key + salt
        for _ in range(100):
            temp_iv = hashlib.sha256(temp_iv).digest()
        return temp_iv[:AES_IV_SIZE]

    @property
    def cipher_initialization(self) -> bytes:
        """Return cipher initialization bytes for the header."""
        return self._salt


class SecureTarRootKeyContext:
    """Handle cipher contexts for multiple inner SecureTar files."""

    def __init__(self, password: str) -> None:
        self._password = password
        self._strategy: KeyDerivationStrategy | None = None
        self._version: int | None = None
        self._derived_keys: dict[Hashable, SecureTarDerivedKeyMaterial] = {}

    @property
    def stream_factory(self) -> type[CipherStreamFactory]:
        """Return the cipher stream factory."""
        if self._strategy is None:
            raise SecureTarError("Context not initialized")
        return self._strategy.stream_factory

    def _ensure_strategy(
        self, version: int, header: SecureTarHeader | None = None
    ) -> KeyDerivationStrategy:
        """Ensure strategy is initialized for the given version."""
        if self._strategy is not None:
            if self._version != version:
                raise SecureTarError(
                    f"Context already initialized for version {self._version}, "
                    f"cannot use for version {version}"
                )
            return self._strategy

        if version in (1, 2):
            self._strategy = KeyDerivationV2(self._password)
        else:
            raise ValueError(f"Unsupported SecureTar version: {version}")

        self._version = version
        return self._strategy

    def derive_key_material(
        self, key_id: Hashable | None, version: int
    ) -> SecureTarDerivedKeyMaterial:
        """Derive per-entry key material from the root key."""
        strategy = self._ensure_strategy(version)
        if key_id is None:
            return strategy.get_key_material()
        if key_id not in self._derived_keys:
            self._derived_keys[key_id] = strategy.get_key_material()
        return self._derived_keys[key_id]

    def restore_key_material(
        self, header: SecureTarHeader
    ) -> SecureTarDerivedKeyMaterial:
        """Reconstruct key material from existing header fields."""
        strategy = self._ensure_strategy(header.version, header)
        return strategy.get_key_material(header.cipher_initialization)


def secure_path(tar: tarfile.TarFile) -> Generator[tarfile.TarInfo, None, None]:
    """Security safe check of path.
    Prevent ../ or absolut paths
    """
    for member in tar:
        file_path = Path(member.name)
        try:
            if file_path.is_absolute():
                raise ValueError()
            Path("/fake", file_path).resolve().relative_to("/fake")
        except (ValueError, RuntimeError):
            _LOGGER.warning("Found issue with file %s", file_path)
            continue
        else:
            yield member


def atomic_contents_add(
    tar_file: tarfile.TarFile,
    origin_path: Path,
    file_filter: Callable[[PurePath], bool],
    arcname: str = ".",
) -> None:
    """Append directories and/or files to the TarFile if file_filter returns False.

    :param file_filter: A filter function, should return True if the item should
    be excluded from the archive. The function should take a single argument, a
    pathlib.PurePath object representing the relative path of the item to be archived.
    """

    if file_filter(PurePath(arcname)):
        return None
    return _atomic_contents_add(tar_file, origin_path, file_filter, arcname)


def _atomic_contents_add(
    tar_file: tarfile.TarFile,
    origin_path: Path,
    file_filter: Callable[[PurePath], bool],
    arcname: str,
) -> None:
    """Append directories and/or files to the TarFile if file_filter returns False."""

    # Add directory only (recursive=False) to ensure we also archive empty directories
    try:
        tar_file.add(origin_path.as_posix(), arcname=arcname, recursive=False)
    except (OSError, tarfile.TarError) as err:
        raise AddFileError(
            origin_path,
            f"Error adding {origin_path} to tarfile: {err} ({err.__class__.__name__})",
        ) from err

    try:
        dir_iterator = origin_path.iterdir()
    except OSError as err:
        raise AddFileError(
            origin_path,
            f"Error iterating over {origin_path}: {err} ({err.__class__.__name__})",
        ) from err

    for directory_item in dir_iterator:
        try:
            item_arcpath = PurePath(arcname, directory_item.name)
            if file_filter(PurePath(item_arcpath)):
                continue

            item_arcname = item_arcpath.as_posix()
            if directory_item.is_dir() and not directory_item.is_symlink():
                _atomic_contents_add(
                    tar_file, directory_item, file_filter, item_arcname
                )
                continue

            tar_file.add(
                directory_item.as_posix(), arcname=item_arcname, recursive=False
            )
        except (OSError, tarfile.TarError) as err:
            raise AddFileError(
                directory_item,
                (
                    f"Error adding {directory_item} to tarfile: "
                    f"{err} ({err.__class__.__name__})"
                ),
            ) from err

    return None
