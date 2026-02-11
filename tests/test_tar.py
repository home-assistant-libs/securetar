"""Test Tarfile functions."""

from collections.abc import Callable, Hashable
import gzip
import io
import os
import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any
from unittest.mock import Mock, patch

import nacl
import pytest

from securetar import (
    SECURETAR_MAGIC,
    SECURETAR_V2_HEADER_SIZE,
    SECURETAR_V3_HEADER_SIZE,
    V3_SECRETSTREAM_ABYTES,
    V3_SECRETSTREAM_CHUNK_SIZE,
    AddFileError,
    InvalidPasswordError,
    SecureTarArchive,
    SecureTarError,
    SecureTarFile,
    SecureTarHeader,
    SecureTarReadError,
    SecureTarRootKeyContext,
    atomic_contents_add,
    secure_path,
)


def get_ciphertext_size_v2(plaintext_size: int) -> int:
    """Get expected ciphertext size for v2."""
    # Padding to next 16 byte block
    padding = 16 - (plaintext_size % 16)
    return plaintext_size + padding + SECURETAR_V2_HEADER_SIZE


def get_ciphertext_size_v3(plaintext_size: int) -> int:
    """Get expected ciphertext size for v3."""
    num_chunks = (
        plaintext_size + V3_SECRETSTREAM_CHUNK_SIZE - 1
    ) // V3_SECRETSTREAM_CHUNK_SIZE
    if num_chunks == 0:
        num_chunks = 1
    return (
        plaintext_size + num_chunks * V3_SECRETSTREAM_ABYTES + SECURETAR_V3_HEADER_SIZE
    )


get_ciphertext_size: dict[int, Callable[[int], int]] = {
    2: get_ciphertext_size_v2,
    3: get_ciphertext_size_v3,
}


@dataclass
class TarInfo:
    """Fake TarInfo."""

    name: str


def test_secure_path() -> None:
    """Test Secure Path."""
    test_list = [
        TarInfo("test.txt"),
        TarInfo("data/xy.blob"),
        TarInfo("bla/blu/ble"),
        TarInfo("data/../xy.blob"),
    ]
    assert test_list == list(secure_path(test_list))


def test_not_secure_path() -> None:
    """Test Not secure path."""
    test_list = [
        TarInfo("/test.txt"),
        TarInfo("data/../../xy.blob"),
        TarInfo("/bla/blu/ble"),
    ]
    assert [] == list(secure_path(test_list))


@pytest.mark.parametrize(
    ("file_filter", "expected_filter_calls", "expected_tar_items"),
    [
        (
            Mock(return_value=False),
            {
                ".",
                "README.md",
                "large_file",
                "test_symlink",
                "test1",
                "test1/script.sh",
            },
            {
                ".",
                "README.md",
                "large_file",
                "test_symlink",
                "test1",
                "test1/script.sh",
            },
        ),
        (
            Mock(return_value=True),
            {"."},
            set(),
        ),
        (
            Mock(wraps=lambda path: path.name == "README.md"),
            {
                ".",
                "README.md",
                "large_file",
                "test_symlink",
                "test1",
                "test1/script.sh",
            },
            {".", "large_file", "test_symlink", "test1", "test1/script.sh"},
        ),
    ],
)
def test_file_filter(
    tmp_path: Path,
    file_filter: Mock,
    expected_filter_calls: set[str],
    expected_tar_items: set[str],
) -> None:
    """Test exclude filter."""
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(temp_tar, "w") as archive:
        with archive.create_tar("core.tar") as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=file_filter,
                arcname=".",
            )
    paths = [call[1][0] for call in file_filter.mock_calls]
    assert len(paths) == len(expected_filter_calls)
    assert set(paths) == {PurePath(path) for path in expected_filter_calls}

    with SecureTarArchive(temp_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as inner_tar_file_obj:
            with tarfile.open(fileobj=inner_tar_file_obj, mode="r") as inner_tar_file:
                members = {tar_info.name for tar_info in inner_tar_file}
    assert members == expected_tar_items


@pytest.mark.parametrize(
    ("target", "attribute", "expected_error"),
    [
        (
            tarfile.TarFile,
            "addfile",
            r"Error adding {temp_orig} to tarfile: Boom! \(OSError\)",
        ),
        (
            tarfile,
            "copyfileobj",
            r"Error adding {temp_orig}/.+ to tarfile: Boom! \(OSError\)",
        ),
        (
            Path,
            "is_dir",
            r"Error adding {temp_orig}/.+ to tarfile: Boom! \(OSError\)",
        ),
        (
            Path,
            "is_symlink",
            r"Error adding {temp_orig}/.+ to tarfile: Boom! \(OSError\)",
        ),
        (
            Path,
            "iterdir",
            r"Error iterating over {temp_orig}: Boom! \(OSError\)",
        ),
    ],
)
def test_create_with_error(
    tmp_path: Path, target: Any, attribute: str, expected_error: str
) -> None:
    """Test error in atomic_contents_add."""
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(temp_tar, "w") as archive:
        with (
            patch.object(target, attribute, side_effect=OSError("Boom!")),
            pytest.raises(
                AddFileError,
                match=expected_error.format(temp_orig=temp_orig),
            ),
            archive.create_tar("core.tar") as inner_tar_file,
        ):
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )


@pytest.mark.parametrize("bufsize", [333, 10240, 4 * 2**20])
@pytest.mark.parametrize("enable_gzip", [True, False])
@pytest.mark.parametrize("version", [2, 3])
def test_create_encrypted_tar_validate(
    tmp_path: Path, bufsize: int, enable_gzip: bool, version: int
) -> None:
    """Test to create a tar file with encryption."""
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)
    with open(temp_orig / "randbytes1", "wb") as file:
        file.write(os.urandom(12345))
    with open(temp_orig / "randbytes2", "wb") as file:
        file.write(os.urandom(12345))

    # Create Tarfile
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        temp_tar,
        "w",
        password=password,
        bufsize=bufsize,
        create_version=version,
    ) as archive:
        with archive.create_tar("core.tar", gzip=enable_gzip) as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )

    assert temp_tar.exists()

    # Attempt to validate password with wrong password
    with SecureTarArchive(temp_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as inner_tar_file_obj:
            secure_tar_file = SecureTarFile(
                None,
                "r",
                bufsize=bufsize,
                fileobj=inner_tar_file_obj,
                password="wrong_password",
                gzip=enable_gzip,
            )
            assert not secure_tar_file.validate_password()

    # Attempt to validate password with correct password
    with SecureTarArchive(temp_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as inner_tar_file_obj:
            secure_tar_file = SecureTarFile(
                None,
                "r",
                bufsize=bufsize,
                fileobj=inner_tar_file_obj,
                password=password,
                gzip=enable_gzip,
            )
            assert secure_tar_file.validate_password()

    # Attempt to validate with wrong password
    with SecureTarArchive(temp_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as inner_tar_file_obj:
            secure_tar_file = SecureTarFile(
                None,
                "r",
                bufsize=bufsize,
                fileobj=inner_tar_file_obj,
                password="wrong_password",
                gzip=enable_gzip,
            )
            assert not secure_tar_file.validate()

    # Attempt to validate with correct password
    with SecureTarArchive(temp_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as inner_tar_file_obj:
            secure_tar_file = SecureTarFile(
                None,
                "r",
                bufsize=bufsize,
                fileobj=inner_tar_file_obj,
                password=password,
                gzip=enable_gzip,
            )
            assert secure_tar_file.validate()


@patch("securetar.time.time", new=Mock(return_value=1765362043.0))
@pytest.mark.parametrize(
    ("derived_key_id", "root_key_context_func", "password", "expect_same_content"),
    [
        (None, lambda: None, "hunter2", False),
        ("inner_file", lambda: SecureTarRootKeyContext("hunter2"), None, True),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_create_encrypted_archive_fixed_nonce(
    tmp_path: Path,
    derived_key_id: Hashable | None,
    root_key_context_func: Callable[[str | None], SecureTarRootKeyContext],
    password: str | None,
    expect_same_content: bool,
    version: int,
) -> None:
    """Test to create an archive with fixed nonce."""
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)
    with open(temp_orig / "randbytes1", "wb") as file:
        file.write(os.urandom(12345))
    with open(temp_orig / "randbytes2", "wb") as file:
        file.write(os.urandom(12345))

    root_key_context = root_key_context_func()

    # Create Archive 1
    temp_tar1 = tmp_path.joinpath("backup1.tar")
    with SecureTarArchive(
        temp_tar1,
        "w",
        create_version=version,
        password=password,
        root_key_context=root_key_context,
    ) as archive:
        with archive.create_tar(
            "core.tar", derived_key_id=derived_key_id
        ) as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )

    # Create Archive 2
    temp_tar2 = tmp_path.joinpath("backup2.tar")
    with SecureTarArchive(
        temp_tar2,
        "w",
        create_version=version,
        password=password,
        root_key_context=root_key_context,
    ) as archive:
        with archive.create_tar(
            "core.tar", derived_key_id=derived_key_id
        ) as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )

    assert expect_same_content == (temp_tar1.read_bytes() == temp_tar2.read_bytes())


@patch("securetar.time.time", new=Mock(return_value=1765362043.0))
@pytest.mark.parametrize(
    ("derived_key_id", "root_key_context_func", "password", "expect_same_content"),
    [
        (None, lambda: None, "hunter2", False),
        ("inner_file", lambda: SecureTarRootKeyContext("hunter2"), None, True),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_encrypt_archive_fixed_nonce(
    tmp_path: Path,
    derived_key_id: Hashable | None,
    root_key_context_func: Callable[[str | None], SecureTarRootKeyContext],
    password: str | None,
    expect_same_content: bool,
    version: int,
) -> None:
    """Test to encrypt a plaintext archive with fixed nonce."""
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)
    with open(temp_orig / "randbytes1", "wb") as file:
        file.write(os.urandom(12345))
    with open(temp_orig / "randbytes2", "wb") as file:
        file.write(os.urandom(12345))

    # Create plaintext archive to encrypt from
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        temp_tar,
        "w",
    ) as archive:
        with archive.create_tar("core.tar") as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )

    root_key_context = root_key_context_func()

    # Create encrypted archive 1
    temp_tar1 = tmp_path.joinpath("backup1.tar")
    with (
        SecureTarArchive(
            temp_tar1,
            "w",
            create_version=version,
            password=password,
            root_key_context=root_key_context,
        ) as encrypted_archive,
        SecureTarArchive(
            temp_tar,
            "r",
        ) as plaintext_archive,
    ):
        for tar_info in plaintext_archive.tar:
            encrypted_archive.import_tar(
                plaintext_archive.tar.extractfile(tar_info),
                tar_info,
                derived_key_id=derived_key_id,
            )

    # Create encrypted archive 2
    temp_tar2 = tmp_path.joinpath("backup2.tar")
    with (
        SecureTarArchive(
            temp_tar2,
            "w",
            create_version=version,
            password=password,
            root_key_context=root_key_context,
        ) as encrypted_archive,
        SecureTarArchive(
            temp_tar,
            "r",
        ) as plaintext_archive,
    ):
        for tar_info in plaintext_archive.tar:
            encrypted_archive.import_tar(
                plaintext_archive.tar.extractfile(tar_info),
                tar_info,
                derived_key_id=derived_key_id,
            )

    assert expect_same_content == (temp_tar1.read_bytes() == temp_tar2.read_bytes())


@pytest.mark.parametrize(
    ("enable_gzip", "inner_tar_files"),
    [
        (True, ("core.tar.gz", "core2.tar.gz", "core3.tar.gz")),
        (False, ("core.tar", "core2.tar", "core3.tar")),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_tar_inside_tar(
    tmp_path: Path, enable_gzip: bool, inner_tar_files: tuple[str, ...], version: int
) -> None:
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        main_tar, "w", create_version=version
    ) as outer_secure_tar_archive:
        for inner_tar_file in inner_tar_files:
            with outer_secure_tar_archive.create_tar(
                inner_tar_file, gzip=enable_gzip
            ) as inner_tar_file:
                atomic_contents_add(
                    inner_tar_file,
                    temp_orig,
                    file_filter=lambda _: False,
                    arcname=".",
                )

        assert len(outer_secure_tar_archive.tar.getmembers()) == 3

        raw_bytes = b'{"test": "test"}'
        fileobj = io.BytesIO(raw_bytes)
        tar_info = tarfile.TarInfo(name="backup.json")
        tar_info.size = len(raw_bytes)
        tar_info.mtime = time.time()
        outer_secure_tar_archive.tar.addfile(tar_info, fileobj=fileobj)
        assert len(outer_secure_tar_archive.tar.getmembers()) == 4

    assert main_tar.exists()

    # Iterate over the tar file, and check there's no securetar header
    files = set()
    with SecureTarFile(main_tar, "r", gzip=False) as tar_file:
        for tar_info in tar_file:
            inner_tar = tar_file.extractfile(tar_info)
            assert inner_tar.read(len(SECURETAR_MAGIC)) != SECURETAR_MAGIC
            files.add(tar_info.name)
    assert files == {"backup.json", *inner_tar_files}

    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(main_tar, "r", gzip=False) as tar_file:
        tar_file.extractall(path=temp_new)

    assert temp_new.is_dir()
    core_tar = temp_new.joinpath(inner_tar_files[0])
    assert core_tar.is_file()
    if enable_gzip:
        compressed = core_tar.read_bytes()
        uncompressed = gzip.decompress(core_tar.read_bytes())
        assert len(uncompressed) > len(compressed)

    assert temp_new.joinpath(inner_tar_files[1]).is_file()
    assert temp_new.joinpath(inner_tar_files[2]).is_file()
    backup_json = temp_new.joinpath("backup.json")
    assert backup_json.is_file()
    assert backup_json.read_bytes() == raw_bytes

    # Extract inner tars
    for inner_tar_file in inner_tar_files:
        temp_inner_new = tmp_path.joinpath(f"{inner_tar_file}_inner_new")

        with SecureTarFile(
            temp_new.joinpath(inner_tar_file), "r", gzip=enable_gzip
        ) as tar_file:
            tar_file.extractall(path=temp_inner_new, members=tar_file)

        assert temp_inner_new.is_dir()
        assert temp_inner_new.joinpath("test_symlink").is_symlink()
        assert temp_inner_new.joinpath("test1").is_dir()
        assert temp_inner_new.joinpath("test1/script.sh").is_file()

        # 775 is correct for local, but in GitHub action it's 755, both is fine
        assert oct(temp_inner_new.joinpath("test1/script.sh").stat().st_mode)[-3:] in [
            "755",
            "775",
        ]
        assert temp_inner_new.joinpath("README.md").is_file()


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("enable_gzip", "inner_tar_files"),
    [
        (True, ("core.tar.gz", "core2.tar.gz", "core3.tar.gz")),
        (False, ("core.tar", "core2.tar", "core3.tar")),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_tar_inside_tar_encrypt(
    tmp_path: Path,
    bufsize: int,
    enable_gzip: bool,
    inner_tar_files: tuple[str, ...],
    version: int,
) -> None:
    """Test we can make encrypted versions of plaintext tars."""

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create an archive with plaintext inner tars
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(main_tar, "w") as outer_secure_tar_archive:
        for inner_tar_file in inner_tar_files:
            with outer_secure_tar_archive.create_tar(
                inner_tar_file, gzip=enable_gzip
            ) as inner_tar_file:
                atomic_contents_add(
                    inner_tar_file,
                    temp_orig,
                    file_filter=lambda _: False,
                    arcname=".",
                )

        assert len(outer_secure_tar_archive.tar.getmembers()) == 3

        raw_bytes = b'{"test": "test"}'
        fileobj = io.BytesIO(raw_bytes)
        tar_info = tarfile.TarInfo(name="backup.json")
        tar_info.size = len(raw_bytes)
        tar_info.mtime = time.time()
        outer_secure_tar_archive.tar.addfile(tar_info, fileobj=fileobj)
        assert len(outer_secure_tar_archive.tar.getmembers()) == 4

    assert main_tar.exists()

    # Iterate over the archive, and check there are no securetar headers in
    # the inner tars
    files = set()
    with SecureTarArchive(main_tar, "r") as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar = outer_secure_tar_archive.tar.extractfile(tar_info)
            assert inner_tar.read(len(SECURETAR_MAGIC)) != SECURETAR_MAGIC
            files.add(tar_info.name)
    assert files == {"backup.json", *inner_tar_files}

    # Create an archive with encrypted inner tars, streamed from the archive
    # with plaintext inner tars
    password = "hunter2"
    temp_encrypted = tmp_path.joinpath("encrypted")
    main_tar_encrypted = temp_encrypted.joinpath("backup.tar")
    os.makedirs(temp_encrypted, exist_ok=True)
    with (
        SecureTarArchive(
            main_tar_encrypted,
            mode="w",
            password=password,
            bufsize=bufsize,
            create_version=version,
            streaming=True,
        ) as encrypted_archive,
        SecureTarArchive(
            main_tar, "r", bufsize=bufsize, streaming=True
        ) as plain_archive,
    ):
        for tar_info in plain_archive.tar:
            encrypted_archive.import_tar(
                plain_archive.tar.extractfile(tar_info), tar_info
            )

    # Check size of encrypted inner tars
    with (
        SecureTarArchive(
            main_tar_encrypted, mode="r", password=password, bufsize=bufsize
        ) as encrypted_archive,
        SecureTarArchive(main_tar, "r", bufsize=bufsize) as plain_archive,
    ):
        for inner_tar_file in inner_tar_files:
            encrypted_tar_info = encrypted_archive.tar.getmember(inner_tar_file)
            plain_tar_info = plain_archive.tar.getmember(inner_tar_file)
            assert encrypted_tar_info.size == get_ciphertext_size[version](
                plain_tar_info.size
            )

    # Check the encrypted inner tars can be opened
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with (
        SecureTarArchive(main_tar_encrypted, password=password) as encrypted_archive,
        SecureTarArchive(main_tar, "r", bufsize=bufsize) as plain_archive,
    ):
        for inner_tar_file in inner_tar_files:
            encrypted_tar_info = encrypted_archive.tar.getmember(inner_tar_file)
            with encrypted_archive.extract_tar(encrypted_tar_info) as decrypted:
                # Check DecryptReader.plaintext_size is correct
                assert (
                    decrypted.plaintext_size
                    == plain_archive.tar.getmember(inner_tar_file).size
                )
                decrypted_inner_tar_path = temp_decrypted.joinpath(inner_tar_file)
                with open(decrypted_inner_tar_path, "wb") as file:
                    while data := decrypted.read(bufsize):
                        file.write(data)

            # Check decrypted file is valid gzip, this fails if the padding is not
            # handled correctly
            if enable_gzip:
                assert decrypted_inner_tar_path.stat().st_size > 0
                gzip.decompress(decrypted_inner_tar_path.read_bytes())

            # Check the tar file can be opened and iterate over it
            files = set()
            with tarfile.open(decrypted_inner_tar_path, "r") as itf:
                for tar_info in itf:
                    files.add(tar_info.name)
            assert files == {
                ".",
                "README.md",
                "large_file",
                "test1",
                "test1/script.sh",
                "test_symlink",
            }


def test_gzipped_tar_inside_tar_failure(tmp_path: Path) -> None:
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(main_tar, "w") as outer_secure_tar_archive:
        # Make the first tar file to ensure that
        # the second tar file can still be created
        with pytest.raises(ValueError, match="Test"):
            with outer_secure_tar_archive.create_tar(
                "failed.tar.gz", gzip=True
            ) as inner_tar_file:
                raise ValueError("Test")

        with pytest.raises(ValueError, match="Test"):
            with outer_secure_tar_archive.create_tar(
                "good.tar.gz", gzip=True
            ) as inner_tar_file:
                atomic_contents_add(
                    inner_tar_file,
                    temp_orig,
                    file_filter=lambda _: False,
                    arcname=".",
                )
                raise ValueError("Test")

        assert len(outer_secure_tar_archive.tar.getmembers()) == 2

    assert main_tar.exists()
    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(main_tar, "r", gzip=False) as tar_file:
        tar_file.extractall(path=temp_new)

    assert temp_new.is_dir()
    assert temp_new.joinpath("good.tar.gz").is_file()

    failed_path = temp_new.joinpath("failed.tar.gz")
    assert failed_path.is_file()

    # Extract inner tar
    temp_inner_new = tmp_path.joinpath("good.tar.gz_inner_new")

    with SecureTarFile(temp_new.joinpath("good.tar.gz"), "r", gzip=True) as tar_file:
        tar_file.extractall(path=temp_inner_new, members=tar_file)

    assert temp_inner_new.is_dir()
    assert temp_inner_new.joinpath("test_symlink").is_symlink()
    assert temp_inner_new.joinpath("test1").is_dir()
    assert temp_inner_new.joinpath("test1/script.sh").is_file()

    # 775 is correct for local, but in GitHub action it's 755, both is fine
    assert oct(temp_inner_new.joinpath("test1/script.sh").stat().st_mode)[-3:] in [
        "755",
        "775",
    ]
    assert temp_inner_new.joinpath("README.md").is_file()

    # Extract failed inner tar (should not raise but will be empty)
    temp_inner_new = tmp_path.joinpath("failed.tar.gz_inner_new")

    with SecureTarFile(temp_new.joinpath("failed.tar.gz"), "r", gzip=True) as tar_file:
        tar_file.extractall(path=temp_inner_new, members=tar_file)


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("enable_gzip", "inner_tar_files"),
    [
        (True, ("core.tar.gz", "core2.tar.gz", "core3.tar.gz")),
        (False, ("core.tar", "core2.tar", "core3.tar")),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_encrypted_tar_inside_tar(
    tmp_path: Path,
    bufsize: int,
    enable_gzip: bool,
    inner_tar_files: tuple[str, ...],
    version: int,
) -> None:
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create an archive with encrypted inner tars
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        main_tar, "w", bufsize=bufsize, create_version=version, password=password
    ) as outer_secure_tar_archive:
        for inner_tar_file in inner_tar_files:
            with outer_secure_tar_archive.create_tar(
                inner_tar_file, gzip=enable_gzip
            ) as inner_tar_file:
                atomic_contents_add(
                    inner_tar_file,
                    temp_orig,
                    file_filter=lambda _: False,
                    arcname=".",
                )

        assert len(outer_secure_tar_archive.tar.getmembers()) == 3

    assert main_tar.exists()

    # Iterate over the archive
    file_sizes: dict[str, int] = {}
    with SecureTarArchive(main_tar, "r", bufsize=bufsize) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar = outer_secure_tar_archive.tar.extractfile(tar_info)
            assert inner_tar.read(len(SECURETAR_MAGIC)) == SECURETAR_MAGIC
            # Skip version and reserved bytes
            inner_tar.read(7)
            file_sizes[tar_info.name] = int.from_bytes(inner_tar.read(8), "big")
    assert set(file_sizes) == {*inner_tar_files}

    # Attempt to decrypt the inner tars with wrong key
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password="wrong_password", streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar_path = temp_decrypted.joinpath(tar_info.name)
            with open(inner_tar_path, "wb") as file:
                # TODO: Check SecureTarReadError message
                with pytest.raises((SecureTarReadError, InvalidPasswordError)):
                    with outer_secure_tar_archive.extract_tar(tar_info) as decrypted:
                        while data := decrypted.read(bufsize):
                            file.write(data)

    # Decrypt the inner tar
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password=password, streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar_path = temp_decrypted.joinpath(tar_info.name)
            with open(inner_tar_path, "wb") as file:
                with outer_secure_tar_archive.extract_tar(tar_info) as decrypted:
                    # Check DecryptReader.plaintext_size is correct
                    assert decrypted.plaintext_size == file_sizes[tar_info.name]
                    while data := decrypted.read(bufsize):
                        file.write(data)

            # Check the indicated size is correct
            assert inner_tar_path.stat().st_size == file_sizes[tar_info.name]

            # Check decrypted file is valid gzip, this fails if the padding is not
            # discarded correctly
            if enable_gzip:
                assert inner_tar_path.stat().st_size > 0
                gzip.decompress(inner_tar_path.read_bytes())

            # Check the tar file can be opened and iterate over it
            files = set()
            with tarfile.open(inner_tar_path, "r") as inner_tar_file:
                for tar_info in inner_tar_file:
                    files.add(tar_info.name)
            assert files == {
                ".",
                "README.md",
                "large_file",
                "test1",
                "test1/script.sh",
                "test_symlink",
            }

    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(main_tar, "r", gzip=False, bufsize=bufsize) as tar_file:
        tar_file.extractall(path=temp_new)

    assert temp_new.is_dir()
    for inner_tar_file in inner_tar_files:
        assert temp_new.joinpath(inner_tar_file).is_file()

    # Extract inner encrypted tars
    for inner_tar_file in inner_tar_files:
        temp_inner_new = tmp_path.joinpath(f"{inner_tar_file}_inner_new")

        with SecureTarFile(
            temp_new.joinpath(inner_tar_file),
            "r",
            password=password,
            gzip=enable_gzip,
            bufsize=bufsize,
        ) as tar_file:
            tar_file.extractall(path=temp_inner_new, members=tar_file)

        assert temp_inner_new.is_dir()
        assert temp_inner_new.joinpath("test_symlink").is_symlink()
        assert temp_inner_new.joinpath("test1").is_dir()
        assert temp_inner_new.joinpath("test1/script.sh").is_file()

        # 775 is correct for local, but in GitHub action it's 755, both is fine
        assert oct(temp_inner_new.joinpath("test1/script.sh").stat().st_mode)[-3:] in [
            "755",
            "775",
        ]
        assert temp_inner_new.joinpath("README.md").is_file()


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("enable_gzip", "inner_tar_files"),
    [
        (True, ("core.tar.gz", "core2.tar.gz", "core3.tar.gz")),
        (False, ("core.tar", "core2.tar", "core3.tar")),
    ],
)
@pytest.mark.parametrize("version", [2, 3])
def test_encrypted_tar_inside_tar_validate(
    tmp_path: Path,
    bufsize: int,
    enable_gzip: bool,
    inner_tar_files: tuple[str, ...],
    version: int,
) -> None:
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create an archive with encrypted inner tars
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        main_tar, "w", bufsize=bufsize, create_version=version, password=password
    ) as outer_secure_tar_archive:
        for inner_tar_file in inner_tar_files:
            with outer_secure_tar_archive.create_tar(
                inner_tar_file, gzip=enable_gzip
            ) as inner_tar_file:
                atomic_contents_add(
                    inner_tar_file,
                    temp_orig,
                    file_filter=lambda _: False,
                    arcname=".",
                )

        assert len(outer_secure_tar_archive.tar.getmembers()) == 3

    assert main_tar.exists()

    # Attempt to validate the inner tars with wrong password
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password="wrong_password", streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert not outer_secure_tar_archive.validate_password(tar_info)

    # Attempt to validate the inner tars with correct password
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password=password, streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert outer_secure_tar_archive.validate_password(tar_info)

    # Attempt to validate the inner tars with wrong password
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password="wrong_password", streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert not outer_secure_tar_archive.validate(tar_info)

    # Attempt to validate the inner tars with correct password
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password=password, streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert outer_secure_tar_archive.validate(tar_info)


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("main_tar_file", "password_validation_result"),
    [
        # Files where the beginning of the file is correct, but there's a
        # secretstream error later in the file. We expect password validation
        # to succeed.
        ("backup_no_final_tag.tar", True),
        ("backup_truncated.tar", True),
        # Files where there's a secretstream error already in the first block,
        # we expect password validation to fail.
        ("backup_early_final_tag.tar", False),
        ("backup_empty.tar", False),
    ],
)
def test_encrypted_tar_inside_tar_validate_secretstream_errors(
    bufsize: int,
    main_tar_file: str,
    password_validation_result: bool,
) -> None:
    password = "hunter2"

    fixture_path = Path(__file__).parent.joinpath("fixtures")
    main_tar = fixture_path.joinpath(f"./{main_tar_file}")

    # Attempt to validate the inner tars with wrong password, we always expect
    # this to fail
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password="wrong_password", streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert not outer_secure_tar_archive.validate_password(tar_info)

    # Attempt to validate the inner tars with correct password
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password=password, streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert (
                outer_secure_tar_archive.validate_password(tar_info)
                == password_validation_result
            )

    # Attempt to validate the inner tars with wrong password, we always expect
    # this to fail
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password="wrong_password", streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert not outer_secure_tar_archive.validate(tar_info)

    # Attempt to validate the inner tars with correct password. All the fixtures
    # have some kind of secretstream error, so we expect validation to fail for
    # all of them.
    with SecureTarArchive(
        main_tar, "r", bufsize=bufsize, password=password, streaming=True
    ) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            assert not outer_secure_tar_archive.validate(tar_info)


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
def test_encrypted_gzipped_tar_inside_tar_legacy_format(
    tmp_path: Path, bufsize: int
) -> None:
    password = "not_correct"

    fixture_path = Path(__file__).parent.joinpath("fixtures")
    main_tar = fixture_path.joinpath("./backup_encrypted_gzipped_legacy_format.tar")

    # Iterate over the tar file, and check there's no securetar header
    files: set[str] = set()
    with SecureTarArchive(main_tar, "r", bufsize=bufsize) as outer_secure_tar_archive:
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar = outer_secure_tar_archive.tar.extractfile(tar_info)
            assert inner_tar.read(len(SECURETAR_MAGIC)) != SECURETAR_MAGIC
            files.add(tar_info.name)
    assert files == {
        "core.tar.gz",
        "core2.tar.gz",
        "core3.tar.gz",
    }

    # Decrypt the inner tar
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with (
        # The fixture was created when passing a key directly, so we mock the key
        patch(
            "securetar.KeyDerivationV2._password_to_key",
            return_value=b"0123456789abcdef",
        ),
        SecureTarArchive(
            main_tar, "r", bufsize=bufsize, password=password
        ) as outer_secure_tar_archive,
    ):
        for tar_info in outer_secure_tar_archive.tar:
            inner_tar_path = temp_decrypted.joinpath(tar_info.name)
            with open(inner_tar_path, "wb") as file:
                with outer_secure_tar_archive.extract_tar(tar_info) as decrypted:
                    while data := decrypted.read(bufsize):
                        file.write(data)

            shutil.copy(inner_tar_path, f"./{inner_tar_path.name}.orig")
            # Rewrite the gzip footer
            # Version 1 of SecureTarFile split the gzip footer in two 16-byte parts,
            # combine them back into a single footer.
            with open(inner_tar_path, "r+b") as file:
                file.seek(-4, io.SEEK_END)
                size_bytes = file.read(4)
                file.seek(-20, io.SEEK_END)
                crc = file.read(4)
                file.seek(-36, io.SEEK_END)
                last_block = file.read(16)
                padding = last_block[-1]
                # Note: This is not a full implementation of the padding removal. Version 1
                # did not add any padding if the inner tar size was a multiple of 16. This
                # means a full implementation needs to try to first treat the file as unpadded.
                # If it fails and the tail is in the range 1..15, it may be padded. Remove
                # the padding and try again. If this also fails, the file is corrupted.
                # In this test case, we only handle the case where the padding is 1..15.
                assert 1 <= padding <= 15
                file.seek(-20 - last_block[-1], io.SEEK_END)
                file.write(crc)
                file.write(size_bytes)
                file.truncate()
            shutil.copy(inner_tar_path, f"./{inner_tar_path.name}.fixed")

            # Check decrypted file is valid gzip, this fails if the padding is not
            # discarded correctly
            assert inner_tar_path.stat().st_size > 0
            gzip.decompress(inner_tar_path.read_bytes())

            # Check the tar file can be opened and iterate over it
            files = set()
            with tarfile.open(inner_tar_path, "r:gz") as inner_tar_file:
                for tar_info in inner_tar_file:
                    files.add(tar_info.name)
            assert files == {
                ".",
                "README.md",
                "test1",
                "test1/script.sh",
                "test_symlink",
            }


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("archive", "inner_tar", "expected_exception", "expected_message"),
    [
        (
            "backup_early_final_tag.tar",
            "core_early_final_tag.tar.gz",
            SecureTarError,
            "Unexpected final tag in secretstream decryption",
        ),
        (
            "backup_no_final_tag.tar",
            "core_no_final_tag.tar.gz",
            SecureTarError,
            "Missing final tag in secretstream decryption",
        ),
        (
            "backup_empty.tar",
            "core_empty.tar.gz",
            nacl.exceptions.ValueError,
            "Ciphertext is too short",
        ),
        (
            "backup_truncated.tar",
            "core_truncated.tar.gz",
            nacl.exceptions.RuntimeError,
            "Unexpected failure",
        ),
    ],
)
def test_archive_secretstream_errors(
    tmp_path: Path,
    bufsize: int,
    archive: str,
    inner_tar: str,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    password = "hunter2"

    fixture_path = Path(__file__).parent.joinpath("fixtures")
    main_tar = fixture_path.joinpath(archive)

    # Attempt decrypting the inner tar
    with (
        SecureTarArchive(
            main_tar, "r", bufsize=bufsize, password=password
        ) as outer_secure_tar_archive,
    ):
        tar_info = outer_secure_tar_archive.tar.getmember(inner_tar)
        with outer_secure_tar_archive.extract_tar(tar_info) as decrypted:
            with pytest.raises(expected_exception, match=expected_message):
                while decrypted.read(bufsize):
                    pass


@pytest.mark.parametrize("bufsize", [33, 333, 10240, 4 * 2**20])
@pytest.mark.parametrize(
    ("tar_name", "expected_exception", "expected_message"),
    [
        (
            "core_early_final_tag.tar.gz",
            SecureTarError,
            "Unexpected final tag in secretstream decryption",
        ),
        (
            "core_no_final_tag.tar.gz",
            SecureTarError,
            "Missing final tag in secretstream decryption",
        ),
        (
            "core_empty.tar.gz",
            nacl.exceptions.ValueError,
            "Ciphertext is too short",
        ),
        (
            "core_truncated.tar.gz",
            nacl.exceptions.RuntimeError,
            "Unexpected failure",
        ),
    ],
)
def test_secretstream_errors(
    tmp_path: Path,
    bufsize: int,
    tar_name: str,
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    password = "hunter2"

    fixture_path = Path(__file__).parent.joinpath("fixtures")
    tar_path = fixture_path.joinpath(tar_name)

    # Attempt decrypting the inner tar
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with pytest.raises(expected_exception, match=expected_message):
        with SecureTarFile(tar_path, "r", bufsize=bufsize, password=password) as tar:
            for tar_info in tar:
                if not tar_info.size:
                    continue
                with tar.extractfile(tar_info) as file:
                    while file.read(bufsize):
                        pass


def test_outer_tar_open_close(tmp_path: Path) -> None:
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")
    outer_secure_tar_archive = SecureTarArchive(main_tar, "w")

    outer_secure_tar_archive.open()
    with outer_secure_tar_archive.create_tar("any.tgz", gzip=True) as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )

    outer_secure_tar_archive.close()

    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(main_tar, "r", gzip=False) as tar_file:
        tar_file.extractall(path=temp_new, members=tar_file)

    assert temp_new.is_dir()
    assert temp_new.joinpath("any.tgz").is_file()


def test_outer_tar_exclusive_mode(tmp_path: Path) -> None:
    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")
    password = "hunter2"
    outer_secure_tar_archive = SecureTarArchive(main_tar, "x", password=password)

    with outer_secure_tar_archive:
        with outer_secure_tar_archive.create_tar("any.tgz", gzip=True):
            pass

    assert main_tar.exists()

    outer_secure_tar_archive = SecureTarArchive(main_tar, "x")
    with pytest.raises(FileExistsError):
        outer_secure_tar_archive.open()


@pytest.mark.parametrize(
    ("params", "expected_exception", "expected_message"),
    [
        (
            {"create_version": 1},
            ValueError,
            "Version must be None when reading a SecureTar file",
        ),
        (
            {"create_version": 1, "mode": "w"},
            ValueError,
            "Unsupported SecureTar version: 1",
        ),
        (
            {"create_version": 4, "mode": "w"},
            ValueError,
            "Unsupported SecureTar version: 4",
        ),
        (
            {"password": "hunter2", "root_key_context": SecureTarRootKeyContext("abc")},
            ValueError,
            "Cannot specify both 'root_key_context' and 'password'",
        ),
        (
            {},
            ValueError,
            "Either name or fileobj must be provided",
        ),
        (
            {"name": "test.tar", "mode": "invalid_mode"},
            ValueError,
            "Mode must be 'x', 'r', or 'w'",
        ),
    ],
)
def test_securetararchive_error_handling(
    params: dict[str, Any],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """Test SecureTarArchive constructor error handling."""
    with pytest.raises(expected_exception, match=expected_message):
        SecureTarArchive(**params)


def test_securetararchive_tar_before_open() -> None:
    """Test SecureTarArchive.tar access before open."""
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="r")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.tar


def test_securetararchive_create_inner_tar_before_open() -> None:
    """Test SecureTarArchive.create_tar call before open."""
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.create_tar("any.tgz")


def test_securetararchive_create_inner_tar_read_mode() -> None:
    """Test SecureTarArchive.create_tar in read mode."""
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with secure_tar_archive:
        pass
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="r")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Archive not open for writing"):
            secure_tar_archive.create_tar("any.tgz")


def test_securetararchive_create_inner_tar_streaming(tmp_path: Path) -> None:
    """Test SecureTarArchive.create_tar in streaming mode."""
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w", streaming=True)
    with secure_tar_archive:
        with pytest.raises(
            SecureTarError, match="create_tar not supported in streaming mode"
        ):
            secure_tar_archive.create_tar("any.tgz")


def test_securetararchive_create_inner_tar_derived_key_unencrypted(
    tmp_path: Path,
) -> None:
    """Test SecureTarArchive.create_tar specify derived key without encryption."""
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w")
    with secure_tar_archive:
        with pytest.raises(
            ValueError,
            match="Cannot specify 'derived_key_id' when encryption is disabled",
        ):
            secure_tar_archive.create_tar("any.tgz", derived_key_id="123")


def test_securetararchive_extract_inner_tar_before_open() -> None:
    """Test SecureTarArchive.extract_tar call before open."""
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.extract_tar(tarfile.TarInfo("blah"))


def test_securetararchive_extract_inner_tar_write_mode() -> None:
    """Test SecureTarArchive.extract_tar in write mode."""
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Archive not open for reading"):
            secure_tar_archive.extract_tar(tarfile.TarInfo("blah"))


def test_securetararchive_extract_inner_tar_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarArchive.extract_tar without encryption."""
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w")
    with secure_tar_archive:
        pass
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="r")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="No password provided"):
            secure_tar_archive.extract_tar(tarfile.TarInfo("blah"))


def test_securetararchive_extract_non_regular_inner_tar(tmp_path: Path) -> None:
    """Test SecureTarArchive.extract_tar with unknown inner tar."""
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w", password="hunter2")
    tarinfo = tarfile.TarInfo("blah")
    tarinfo.type = tarfile.DIRTYPE
    with secure_tar_archive:
        secure_tar_archive.tar.addfile(tarinfo)
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="r", password="hunter2")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Cannot extract blah"):
            secure_tar_archive.extract_tar(tarinfo)


def test_securetararchive_import_tar_before_open() -> None:
    """Test SecureTarArchive.import_tar call before open."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.import_tar(Mock(), tarinfo)


def test_securetararchive_import_tar_read_mode() -> None:
    """Test SecureTarArchive.import_tar in read mode."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with secure_tar_archive:
        pass
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="r")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Archive not open for writing"):
            secure_tar_archive.import_tar(Mock(), tarinfo)


def test_securetararchive_import_tar_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarArchive.import_tar specify derived key without encryption."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="No password provided"):
            secure_tar_archive.import_tar(Mock(), tarinfo)


def test_securetararchive_validate_password_before_open() -> None:
    """Test SecureTarArchive.validate_password call before open."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.validate_password(tarinfo)


def test_securetararchive_validate_password_write_mode() -> None:
    """Test SecureTarArchive.validate_password in write mode."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Archive not open for reading"):
            secure_tar_archive.validate_password(tarinfo)


def test_securetararchive_validate_password_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarArchive.validate_password specify derived key without encryption."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w")
    with secure_tar_archive:
        pass
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="r")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="No password provided"):
            secure_tar_archive.validate_password(tarinfo)


def test_securetararchive_validate_before_open() -> None:
    """Test SecureTarArchive.validate call before open."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with pytest.raises(SecureTarError, match="Archive not open"):
        secure_tar_archive.validate(tarinfo)


def test_securetararchive_validate_write_mode() -> None:
    """Test SecureTarArchive.validate in write mode."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    secure_tar_archive = SecureTarArchive(name=Path("test.tar"), mode="w")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="Archive not open for reading"):
            secure_tar_archive.validate(tarinfo)


def test_securetararchive_validate_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarArchive.validate specify derived key without encryption."""
    tarinfo = tarfile.TarInfo("any.tgz")
    tarinfo.size = 1234
    main_tar = tmp_path.joinpath("test.tar")
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="w")
    with secure_tar_archive:
        pass
    secure_tar_archive = SecureTarArchive(name=main_tar, mode="r")
    with secure_tar_archive:
        with pytest.raises(SecureTarError, match="No password provided"):
            secure_tar_archive.validate(tarinfo)


@pytest.mark.parametrize(
    ("params", "expected_exception", "expected_message"),
    [
        (
            {"create_version": 1},
            ValueError,
            "Version must be None when reading a SecureTar file",
        ),
        (
            {"create_version": 1, "mode": "w"},
            ValueError,
            "Unsupported SecureTar version: 1",
        ),
        (
            {"create_version": 4, "mode": "w"},
            ValueError,
            "Unsupported SecureTar version: 4",
        ),
        (
            {"derived_key_id": "123"},
            ValueError,
            "Cannot specify 'derived_key_id' without 'root_key_context'",
        ),
        (
            {"password": "hunter2", "root_key_context": SecureTarRootKeyContext("abc")},
            ValueError,
            "Cannot specify both 'root_key_context' and 'password'",
        ),
        (
            {},
            ValueError,
            "Either filename or fileobj must be provided",
        ),
        (
            {"name": "test.tar", "mode": "invalid_mode"},
            ValueError,
            "Mode must be 'r'",
        ),
    ],
)
def test_securetarfile_error_handling(
    params: dict[str, Any],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    """Test SecureTarFile constructor error handling."""
    with pytest.raises(expected_exception, match=expected_message):
        SecureTarFile(**params)


def test_securetarfile_validate_password_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarFile.validate_password specify derived key without encryption."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w") as archive:
        with archive.create_tar("core.tar"):
            pass
    with SecureTarArchive(main_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as fileobj:
            secure_tar_file = SecureTarFile(fileobj=fileobj, mode="r")
            with pytest.raises(SecureTarError, match="File is not encrypted"):
                secure_tar_file.validate_password()


def test_securetarfile_validate_password_write_mode(tmp_path: Path) -> None:
    """Test SecureTarFile.validate_password in write mode."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w", password="hunter2") as archive:
        inner_tar = archive.create_tar("core.tar")
        with pytest.raises(
            SecureTarError, match="Can only validate password in read mode"
        ):
            inner_tar.validate_password()


def test_securetarfile_validate_password_after_open(tmp_path: Path) -> None:
    """Test SecureTarFile.validate_password call after open."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w", password="hunter2") as archive:
        with archive.create_tar("core.tar"):
            pass
    with SecureTarArchive(main_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as fileobj:
            secure_tar_file = SecureTarFile(
                fileobj=fileobj, mode="r", password="hunter2"
            )
            with secure_tar_file:
                with pytest.raises(SecureTarError, match="File is already open"):
                    secure_tar_file.validate_password()


def test_securetarfile_validate_unencrypted(tmp_path: Path) -> None:
    """Test SecureTarFile.validate specify derived key without encryption."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w") as archive:
        with archive.create_tar("core.tar"):
            pass
    with SecureTarArchive(main_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as fileobj:
            secure_tar_file = SecureTarFile(fileobj=fileobj, mode="r")
            with pytest.raises(SecureTarError, match="File is not encrypted"):
                secure_tar_file.validate()


def test_securetarfile_validate_write_mode(tmp_path: Path) -> None:
    """Test SecureTarFile.validate in write mode."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w", password="hunter2") as archive:
        inner_tar = archive.create_tar("core.tar")
        with pytest.raises(
            SecureTarError, match="Can only validate password in read mode"
        ):
            inner_tar.validate()


def test_securetarfile_validate_after_open(tmp_path: Path) -> None:
    """Test SecureTarFile.validate call after open."""
    main_tar = tmp_path.joinpath("test.tar")
    with SecureTarArchive(main_tar, "w", password="hunter2") as archive:
        with archive.create_tar("core.tar"):
            pass
    with SecureTarArchive(main_tar, "r") as archive:
        with archive.tar.extractfile("core.tar") as fileobj:
            secure_tar_file = SecureTarFile(
                fileobj=fileobj, mode="r", password="hunter2"
            )
            with secure_tar_file:
                with pytest.raises(SecureTarError, match="File is already open"):
                    secure_tar_file.validate()


def test_securetarfile_path_fileobj() -> None:
    """Test SecureTarFile.path property when fileobj is used."""
    secure_tar_file = SecureTarFile(fileobj=io.BytesIO(), mode="r")
    assert secure_tar_file.path is None


def test_securetarfile_path_name() -> None:
    """Test SecureTarFile.path property when name is used."""
    secure_tar_file = SecureTarFile(name=Path("test.tar"), mode="r")
    assert secure_tar_file.path == Path("test.tar")


def test_securetarfile_size_fileobj() -> None:
    """Test SecureTarFile.size property when fileobj is used."""
    secure_tar_file = SecureTarFile(fileobj=io.BytesIO(), mode="r")
    assert secure_tar_file.size == 0


def test_securetarfile_size_name() -> None:
    """Test SecureTarFile.size property when name is used."""
    secure_tar_file = SecureTarFile(name=Path("test.tar"), mode="r")
    assert secure_tar_file.size == 0.01


def test_securetarfile_v1_fallback() -> None:
    """Test SecureTarFile with invalid magic."""
    header_data = b"invalid magicabc"
    header = SecureTarHeader.from_bytes(io.BytesIO(header_data))
    assert header.version == 1
    assert header.cipher_initialization == header_data


@pytest.mark.parametrize(
    ("header", "expected_exception", "expected_message"),
    [
        (
            b"SecureTar\x01\x00\x00\x00\x00\x00\x00",
            ValueError,
            "Unsupported SecureTar version: 1",
        ),
        (
            b"SecureTar\x04\x00\x00\x00\x00\x00\x00",
            ValueError,
            "Unsupported SecureTar version: 4",
        ),
        (
            b"SecureTar\x02\x00\x00\x00\x00\x00\x01",
            ValueError,
            "Invalid reserved bytes in SecureTar header",
        ),
    ],
)
def test_securetarfile_invalid_magic(
    header, expected_exception, expected_message
) -> None:
    """Test SecureTarFile with invalid magic."""
    with pytest.raises(expected_exception, match=expected_message):
        SecureTarHeader.from_bytes(io.BytesIO(header))
