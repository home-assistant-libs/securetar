"""Tarfile fileobject handler for encrypted files."""

from __future__ import annotations

from collections.abc import Callable, Generator, Hashable
import copy
from dataclasses import dataclass
import enum
import hashlib
import logging
import os
import tarfile
import time
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, IO, BinaryIO, Literal

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    CipherContext,
    algorithms,
    modes,
)

_LOGGER: logging.Logger = logging.getLogger(__name__)

BLOCK_SIZE = 16
BLOCK_SIZE_BITS = 128
IV_SIZE = BLOCK_SIZE
DEFAULT_BUFSIZE = 10240

SECURETAR_MAGIC = b"SecureTar\x02\x00\x00\x00\x00\x00\x00"
SECURETAR_HEADER_SIZE = len(SECURETAR_MAGIC) + 16

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

    def __init__(self, cbc_rand: bytes, plaintext_size: int | None) -> None:
        """Initialize SecureTar header."""
        self.cbc_rand = cbc_rand
        self.plaintext_size = plaintext_size

    @classmethod
    def from_bytes(cls, f: IO[bytes]) -> SecureTarHeader:
        """Return header bytes."""
        header = f.read(len(SECURETAR_MAGIC))
        plaintext_size: int | None = None
        if header != SECURETAR_MAGIC:
            cbc_rand = header
        else:
            plaintext_size = int.from_bytes(f.read(8), "big")
            f.read(8)  # Skip reserved bytes
            cbc_rand = f.read(IV_SIZE)

        return cls(cbc_rand, plaintext_size)

    def to_bytes(self) -> bytes:
        """Return header bytes."""
        if self.plaintext_size is None:
            raise ValueError("Plaintext size is required")
        return (
            SECURETAR_MAGIC
            + self.plaintext_size.to_bytes(8, "big")
            + bytes(8)
            + self.cbc_rand
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


class _CipherIO:
    """Base class for SecureTar cipher operations."""

    def __init__(
        self,
        cipher_mode: CipherMode,
        *,
        root_key_context: SecureTarRootKeyContext,
        derived_key_id: Hashable | None,
    ) -> None:
        self._cipher_mode = cipher_mode
        self._root_key_context = root_key_context
        self._derived_key_id = derived_key_id

        self._cipher: CipherContext | None = None
        self._padder: padding.PaddingContext | None = None
        self.securetar_header: SecureTarHeader | None = None
        self.padding_length = 0

    def _open_for_decrypt(self, source: IO[bytes]) -> None:
        """Initialize cipher for decryption."""
        self.securetar_header = SecureTarHeader.from_bytes(source)
        derived_key_material = self._root_key_context.restore_key_material(
            header=self.securetar_header
        )
        self._setup_cipher(derived_key_material)

    def _open_for_encrypt(self, plaintext_size: int | None) -> None:
        """Initialize cipher for encryption."""
        derived_key_material = self._root_key_context.derive_key_material(
            self._derived_key_id
        )
        self.securetar_header = SecureTarHeader(
            derived_key_material.nonce,
            plaintext_size,
        )
        self._setup_cipher(derived_key_material)

    def _setup_cipher(self, derived_key_material: SecureTarDerivedKeyMaterial) -> None:
        key = derived_key_material.key
        iv = derived_key_material.iv
        aes = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        if self._cipher_mode == CipherMode.DECRYPT:
            self._cipher = aes.decryptor()
        else:
            self._cipher = aes.encryptor()
        self._padder = padding.PKCS7(BLOCK_SIZE_BITS).padder()

    def finalize_padding(self) -> bytes:
        """Return final padding block.

        Can only be called once.
        """
        if not self._cipher or not self._padder:
            raise SecureTarError("Cipher not initialized")
        if self._cipher_mode == CipherMode.ENCRYPT:
            padding_data = self._padder.finalize()
            self.padding_length = len(padding_data)
            self._padder = None  # Mark as finalized
            return self._cipher.update(padding_data)
        return b""

    def close(self) -> None:
        """Mark as closed."""
        self._cipher = None
        self._padder = None


class _CipherReader(_CipherIO):
    """Reads from a stream and encrypts or decrypts."""

    def __init__(
        self,
        source: IO[bytes],
        cipher_mode: CipherMode,
        *,
        root_key_context: SecureTarRootKeyContext,
        derived_key_id: Hashable | None = None,
    ) -> None:
        super().__init__(
            cipher_mode,
            root_key_context=root_key_context,
            derived_key_id=derived_key_id,
        )
        self._source = source

    def open(self, plaintext_size: int | None = None) -> None:
        """Initialize cipher."""
        if self._cipher_mode == CipherMode.DECRYPT:
            self._open_for_decrypt(self._source)
        else:
            if plaintext_size is None:
                raise ValueError("plaintext_size required for encryption")
            self._open_for_encrypt(plaintext_size)

    def read(self, size: int = 0) -> bytes:
        """Read from source and transform (encrypt or decrypt)."""
        if not self._cipher or not self._padder:
            raise SecureTarError("Cipher not initialized")
        data = self._padder.update(self._source.read(size))
        return self._cipher.update(data)


class _CipherWriter(_CipherIO):
    """Encrypts data and writes to a stream."""

    def __init__(
        self,
        dest: IO[bytes],
        *,
        root_key_context: SecureTarRootKeyContext,
        derived_key_id: Hashable | None = None,
    ) -> None:
        super().__init__(
            CipherMode.ENCRYPT,
            root_key_context=root_key_context,
            derived_key_id=derived_key_id,
        )
        self._dest = dest

    def open(self) -> None:
        """Initialize cipher and write header."""
        self._open_for_encrypt(0)  # Size unknown at this point
        self._dest.write(self.securetar_header.to_bytes())

    def write(self, data: bytes) -> None:
        """Encrypt and write to destination."""
        if not self._cipher or not self._padder:
            raise SecureTarError("Cipher not initialized")
        data = self._padder.update(data)
        self._dest.write(self._cipher.update(data))

    def finalize_header(self, ciphertext_size: int) -> None:
        """Update header with plaintext size based on final ciphertext size."""
        # The size of the plaintext is the size of the ciphertext size minus:
        # - The size of the SecureTar header if present (secure tar version 2+)
        # - The size of the IV
        # - The size of the padding added (1..16 bytes)
        if self.securetar_header and self.padding_length:
            self.securetar_header.plaintext_size = (
                ciphertext_size - self.padding_length - IV_SIZE - SECURETAR_HEADER_SIZE
            )

    def close(self) -> None:
        """Finalize and write padding."""
        if self._padder:
            padding_bytes = self.finalize_padding()
            self._dest.write(padding_bytes)
        super().close()


class _FramedReader:
    """Base class to decrypt or encrypt a stream with framing."""

    def __init__(
        self,
        source: _CipherReader,
        size: int,
        header: bytes | None = None,
    ) -> None:
        """Initialize."""
        self._head = header
        self._source = source
        self._pos = 0
        self._size = size
        self._tail: bytes | None = None

    def read(self, size: int = 0) -> bytes:
        """Read data."""
        if self._tail is not None:
            # Finish reading tail
            data = self._tail[:size]
            self._tail = self._tail[size:]
            return data

        if self._head:
            # Read from head
            data = self._head[:size]
            self._head = self._head[size:]
            remaining = size - len(data)
            if remaining:
                data += self._source.read(remaining)
        else:
            data = self._source.read(size)

        self._pos += len(data)
        if not data or self._size - self._pos > BLOCK_SIZE:
            return data

        # Last block: Append any remaining head, read tail and handle padding
        if self._head:
            data += self._head
        data += self._source.read(max(self._size - self._pos, 0))
        data = self._process_padding(data)
        self._tail = data[size:]
        return data[:size]

    def _process_padding(self, data: bytes) -> bytes:
        """Process padding."""
        raise NotImplementedError


class _FramedDecryptingReader(_FramedReader):
    """Decrypt a stream with framing."""

    def read(self, size: int = 0) -> bytes:
        """Read data."""
        if self._head is None:
            # Read and validate header
            self._head = self._source.read(max(size, TAR_BLOCK_SIZE))
            if not _is_valid_tar_header(self._head):
                raise SecureTarReadError("The inner tar is not gzip or tar, wrong key?")

        return super().read(size)

    def _process_padding(self, data: bytes) -> bytes:
        """Process padding."""
        padding_len = data[-1]
        return data[:-padding_len]


class _SecureTarDecryptStream:
    """Decrypts an encrypted tar read from a stream."""

    _cipher: _CipherReader

    def __init__(
        self,
        source: IO[bytes],
        *,
        ciphertext_size: int,
        root_key_context: SecureTarRootKeyContext,
    ) -> None:
        self._source = source
        self._root_key_context = root_key_context
        self._ciphertext_size = ciphertext_size

    def __enter__(self) -> _FramedDecryptingReader:
        """Returns a readable stream of decrypted bytes."""
        self._cipher = _CipherReader(
            self._source,
            CipherMode.DECRYPT,
            root_key_context=self._root_key_context,
        )
        self._cipher.open()

        # The size of the plaintext is the ciphertext size minus:
        # - The size of the SecureTar header if present (secure tar version 2+)
        # - The size of the IV
        # - Padding (discarded by _FramedDecryptingReader, not known for version 1)
        plaintext_size_with_padding = self._ciphertext_size - IV_SIZE
        if self._cipher.securetar_header.plaintext_size is not None:
            plaintext_size_with_padding -= SECURETAR_HEADER_SIZE

        return _FramedDecryptingReader(self._cipher, plaintext_size_with_padding)

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self._cipher.close()

    def validate(self) -> bool:
        """Validate the password by checking if decrypted data looks like a tar.

        Note: This consumes the stream. Create a new instance to read data.

        Returns:
            True if password is valid, False otherwise.
        """
        with self as reader:
            try:
                reader.read(1)
                return True
            except SecureTarReadError:
                return False


class _FramedEncryptingReader(_FramedReader):
    """Encrypt a stream with framing."""

    def __init__(
        self,
        source: _CipherReader,
        size: int,
        *,
        header: bytes,
    ) -> None:
        """Initialize."""
        self.encrypted_size = size
        super().__init__(source, size, header=header)

    def _process_padding(self, data: bytes) -> bytes:
        """Process padding."""
        return data + self._source.finalize_padding()


class _SecureTarEncryptStream:
    """Encrypt a plaintext tar read from a stream."""

    _cipher: _CipherReader

    def __init__(
        self,
        source: IO[bytes],
        *,
        derived_key_id: Hashable | None,
        plaintext_size: int,
        root_key_context: SecureTarRootKeyContext,
    ) -> None:
        self._source = source
        self._derived_key_id = derived_key_id
        self._plaintext_size = plaintext_size
        self._root_key_context = root_key_context

    def __enter__(self) -> _FramedEncryptingReader:
        """Returns a readable stream of encrypted bytes."""
        self._cipher = _CipherReader(
            self._source,
            CipherMode.ENCRYPT,
            root_key_context=self._root_key_context,
            derived_key_id=self._derived_key_id,
        )
        self._cipher.open(plaintext_size=self._plaintext_size)

        # The ciphertext size is the sum of:
        # - The size of the SecureTar header
        # - The size of the IV
        # - The size of the plaintext, padded to the next cipher block boundary
        ciphertext_size = (
            SECURETAR_HEADER_SIZE
            + IV_SIZE
            + self._plaintext_size
            + BLOCK_SIZE
            - self._plaintext_size % BLOCK_SIZE
        )

        return _FramedEncryptingReader(
            self._cipher,
            ciphertext_size,
            header=self._cipher.securetar_header.to_bytes(),
        )

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self._cipher.close()


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
        self._cipher: _CipherReader | _CipherWriter | None = None

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
            self._cipher = _CipherReader(
                self._file,
                CipherMode.DECRYPT,
                root_key_context=self._root_key_context,
                derived_key_id=None,
            )
            self._cipher.open()
        else:
            self._cipher = _CipherWriter(
                self._file,
                root_key_context=self._root_key_context,
                derived_key_id=self._derived_key_id,
            )
            self._cipher.open()

        # Open tar with cipher as fileobj
        self._tar = tarfile.open(
            fileobj=self._cipher,
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
        if self._cipher:
            self._cipher.close()
            self._cipher = None
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
            cipher = _CipherReader(
                file,
                CipherMode.DECRYPT,
                root_key_context=self._root_key_context,
            )
            cipher.open()

            data = cipher.read(TAR_BLOCK_SIZE)
            cipher.close()

            return _is_valid_tar_header(data)
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


class _InnerSecureTarFile(SecureTarFile):
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
        self.stream: Generator[BinaryIO, None, None] | None = None

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
        cipher = self._cipher
        super().__exit__(exc_type, exc_value, traceback)
        self._finalize_tar_entry(cipher)

    def _finalize_tar_entry(self, cipher: _CipherWriter | None) -> None:
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
        if self._encrypted and cipher and cipher.padding_length:
            cipher.finalize_header(size_of_inner_tar)
            fileobj.seek(self._header_position + self._header_length)
            fileobj.write(cipher.securetar_header.to_bytes())

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
    ) -> _InnerSecureTarFile:
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
            raise SecureTarError("create_inner_tar not supported in streaming mode")
        if derived_key_id is not None and self._root_key_context is None:
            raise ValueError(
                "Cannot specify 'derived_key_id' when encryption is disabled"
            )

        return _InnerSecureTarFile(
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
    ) -> _SecureTarDecryptStream:
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

        return _SecureTarDecryptStream(
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

        The tar_info.size must be set to the size of the source stream.

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

        with _SecureTarEncryptStream(
            source,
            derived_key_id=derived_key_id,
            plaintext_size=member.size,
            root_key_context=self._root_key_context,
        ) as encrypted:
            encrypted_tar_info = copy.deepcopy(member)
            encrypted_tar_info.size = encrypted.encrypted_size
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


@dataclass(frozen=True, kw_only=True)
class SecureTarDerivedKeyMaterial:
    """Dataclass to hold key and iv for encrypted SecureTar file."""

    key: bytes
    iv: bytes
    nonce: bytes


class SecureTarRootKeyContext:
    """Handle cipher contexts for multiple inner SecureTar files."""

    _key: bytes | None = None

    def __init__(self, password: str):
        """Initialize."""
        self._password = password
        self._derived_keys: dict[Hashable, SecureTarDerivedKeyMaterial] = {}

    def derive_key_material(
        self, key_id: Hashable | None = None
    ) -> SecureTarDerivedKeyMaterial:
        """Derive per-entry key material from the root key."""
        if not self._key:
            self._key = self._password_to_key(self._password)
        if key_id is None:
            return self._derive_key_material()
        if key_id not in self._derived_keys:
            self._derived_keys[key_id] = self._derive_key_material()
        return self._derived_keys[key_id]

    def _derive_key_material(self) -> SecureTarDerivedKeyMaterial:
        """Derive per-entry key material from the root key."""
        cbc_rand = os.urandom(IV_SIZE)
        iv = self._generate_iv(self._key, cbc_rand)
        return SecureTarDerivedKeyMaterial(key=self._key, iv=iv, nonce=cbc_rand)

    def restore_key_material(
        self, header: SecureTarHeader
    ) -> SecureTarDerivedKeyMaterial:
        """Reconstruct key material from existing header fields."""
        if not self._key:
            self._key = self._password_to_key(self._password)
        cbc_rand = header.cbc_rand
        iv = self._generate_iv(self._key, cbc_rand)
        return SecureTarDerivedKeyMaterial(key=self._key, iv=iv, nonce=cbc_rand)

    @staticmethod
    def _password_to_key(password: str) -> bytes:
        """Generate a AES Key from password.

        Uses 100 rounds of SHA256 hashing to derive a 16-byte key from a password.
        """
        key: bytes = password.encode()
        for _ in range(100):
            key = hashlib.sha256(key).digest()
        return key[:16]

    @staticmethod
    def _generate_iv(key: bytes, salt: bytes) -> bytes:
        """Generate an iv from data."""
        temp_iv = key + salt
        for _ in range(100):
            temp_iv = hashlib.sha256(temp_iv).digest()
        return temp_iv[:IV_SIZE]


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
