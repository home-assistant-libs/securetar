"""Test Tarfile functions."""

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

import pytest

from securetar import (
    AddFileError,
    SECURETAR_MAGIC,
    SecureTarArchive,
    SecureTarError,
    SecureTarFile,
    SecureTarReadError,
    SecureTarRootKeyContext,
    atomic_contents_add,
    secure_path,
)


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


def test_file_filter(tmp_path: Path) -> None:
    """Test exclude filter."""
    file_filter = Mock(return_value=False)
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarFile(temp_tar, "w") as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=file_filter,
            arcname=".",
        )
    paths = [call[1][0] for call in file_filter.mock_calls]
    expected_paths = {
        PurePath("."),
        PurePath("README.md"),
        PurePath("test_symlink"),
        PurePath("test1"),
        PurePath("test1/script.sh"),
    }
    assert len(paths) == len(expected_paths)
    assert set(paths) == expected_paths


@pytest.mark.parametrize("bufsize", [10240, 4 * 2**20])
def test_create_pure_tar(tmp_path: Path, bufsize: int) -> None:
    """Test to create a tar file without encryption."""
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
    temp_tar = tmp_path.joinpath("backup.tar")
    with SecureTarFile(temp_tar, "w", bufsize=bufsize) as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )

    assert temp_tar.exists()

    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(temp_tar, "r", bufsize=bufsize) as tar_file:
        tar_file.extractall(path=temp_new, members=tar_file)

    assert temp_new.is_dir()
    assert temp_new.joinpath("test_symlink").is_symlink()
    assert temp_new.joinpath("test1").is_dir()
    assert temp_new.joinpath("test1/script.sh").is_file()

    # 775 is correct for local, but in GitHub action it's 755, both is fine
    assert oct(temp_new.joinpath("test1/script.sh").stat().st_mode)[-3:] in [
        "755",
        "775",
    ]
    assert temp_new.joinpath("README.md").is_file()


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
    with (
        patch.object(target, attribute, side_effect=OSError("Boom!")),
        pytest.raises(
            AddFileError,
            match=expected_error.format(temp_orig=temp_orig),
        ),
        SecureTarFile(temp_tar, "w") as tar_file,
    ):
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )


@pytest.mark.parametrize("bufsize", [333, 10240, 4 * 2**20])
def test_create_encrypted_tar(tmp_path: Path, bufsize: int) -> None:
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
    with SecureTarFile(temp_tar, "w", password=password, bufsize=bufsize) as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )

    assert temp_tar.exists()

    # Iterate over the tar file
    files = set()
    with SecureTarFile(temp_tar, "r", password=password, bufsize=bufsize) as tar_file:
        for tar_info in tar_file:
            files.add(tar_info.name)
    assert files == {
        ".",
        "README.md",
        "randbytes1",
        "randbytes2",
        "test_symlink",
        "test1",
        "test1/script.sh",
    }

    # Restore
    temp_new = tmp_path.joinpath("new")
    with SecureTarFile(temp_tar, "r", password=password, bufsize=bufsize) as tar_file:
        tar_file.extractall(path=temp_new, members=tar_file)

    assert temp_new.is_dir()
    assert temp_new.joinpath("test_symlink").is_symlink()
    assert temp_new.joinpath("test1").is_dir()
    assert temp_new.joinpath("test1/script.sh").is_file()
    assert temp_new.joinpath("randbytes1").is_file()
    assert temp_new.joinpath("randbytes2").is_file()

    # 775 is correct for local, but in GitHub action it's 755, both is fine
    assert oct(temp_new.joinpath("test1/script.sh").stat().st_mode)[-3:] in [
        "755",
        "775",
    ]
    assert temp_new.joinpath("README.md").is_file()


@patch("securetar.time.time", new=Mock(return_value=1765362043.0))
@pytest.mark.parametrize(
    ("create_cipher_context", "expect_same_content"),
    [(False, False), (True, True)],
)
def test_create_encrypted_archive_fixed_nonce(
    tmp_path: Path, create_cipher_context: bool, expect_same_content: bool
) -> None:
    """Test to create an archive with fixed nonce."""
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)
    with open(temp_orig / "randbytes1", "wb") as file:
        file.write(os.urandom(12345))
    with open(temp_orig / "randbytes2", "wb") as file:
        file.write(os.urandom(12345))

    derived_key_id = None
    root_key_context = None
    if create_cipher_context:
        derived_key_id = "inner_file"
        root_key_context = SecureTarRootKeyContext(password)
        password = None

    # Create Archive 1
    temp_tar1 = tmp_path.joinpath("backup1.tar")
    with SecureTarArchive(
        temp_tar1,
        "w",
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
    ("create_cipher_context", "expect_same_content"),
    [(False, False), (True, True)],
)
def test_encrypt_archive_fixed_nonce(
    tmp_path: Path, create_cipher_context: bool, expect_same_content: bool
) -> None:
    """Test to encrypt a plaintext archive with fixed nonce."""
    password = "hunter2"

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
        with archive.create_tar(
            "core.tar"
        ) as inner_tar_file:
            atomic_contents_add(
                inner_tar_file,
                temp_orig,
                file_filter=lambda _: False,
                arcname=".",
            )

    derived_key_id = None
    root_key_context = None
    if create_cipher_context:
        derived_key_id = "inner_file"
        root_key_context = SecureTarRootKeyContext(password)
        password = None

    # Create encrypted archive 1
    temp_tar1 = tmp_path.joinpath("backup1.tar")
    with SecureTarArchive(
        temp_tar1,
        "w",
        password=password,
        root_key_context=root_key_context,
    ) as encrypted_archive, SecureTarArchive(
        temp_tar,
        "r",
    ) as plaintext_archive:
        for tar_info in plaintext_archive.tar:
            encrypted_archive.import_tar(
                plaintext_archive.tar.extractfile(tar_info), tar_info, derived_key_id=derived_key_id
            )

    # Create encrypted archive 2
    temp_tar2 = tmp_path.joinpath("backup2.tar")
    with SecureTarArchive(
        temp_tar2,
        "w",
        password=password,
        root_key_context=root_key_context,
    ) as encrypted_archive, SecureTarArchive(
        temp_tar,
        "r",
    ) as plaintext_archive:
        for tar_info in plaintext_archive.tar:
            encrypted_archive.import_tar(
                plaintext_archive.tar.extractfile(tar_info), tar_info, derived_key_id=derived_key_id
            )

    assert expect_same_content == (temp_tar1.read_bytes() == temp_tar2.read_bytes())


@pytest.mark.parametrize(
    ("create_cipher_context", "expect_same_content"),
    [(False, False), (True, True)],
)
def test_create_encrypted_tar_fixed_nonce(
    tmp_path: Path, create_cipher_context: bool, expect_same_content: bool
) -> None:
    """Test to create a tar file with fixed nonce."""
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)
    with open(temp_orig / "randbytes1", "wb") as file:
        file.write(os.urandom(12345))
    with open(temp_orig / "randbytes2", "wb") as file:
        file.write(os.urandom(12345))

    derived_key_id = None
    root_key_context = None

    # Create Tarfile1
    if create_cipher_context:
        derived_key_id = "inner_file"
        root_key_context = SecureTarRootKeyContext(password)
    temp_tar1 = tmp_path.joinpath("backup1.tar")
    with SecureTarFile(
        temp_tar1,
        "w",
        derived_key_id=derived_key_id,
        root_key_context=root_key_context,
    ) as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )

    # Create Tarfile2
    temp_tar2 = tmp_path.joinpath("backup2.tar")
    with SecureTarFile(
        temp_tar2,
        "w",
        derived_key_id=derived_key_id,
        root_key_context=root_key_context,
    ) as tar_file:
        atomic_contents_add(
            tar_file,
            temp_orig,
            file_filter=lambda _: False,
            arcname=".",
        )

    same_content = temp_tar1.read_bytes() == temp_tar2.read_bytes()
    assert same_content == expect_same_content


@pytest.mark.parametrize(
    ("enable_gzip", "inner_tar_files"),
    [
        (True, ("core.tar.gz", "core2.tar.gz", "core3.tar.gz")),
        (False, ("core.tar", "core2.tar", "core3.tar")),
    ],
)
def test_tar_inside_tar(
    tmp_path: Path, enable_gzip: bool, inner_tar_files: tuple[str, ...]
) -> None:
    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create Tarfile
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
def test_tar_inside_tar_encrypt(
    tmp_path: Path, bufsize: int, enable_gzip: bool, inner_tar_files: tuple[str, ...]
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
            assert (
                encrypted_tar_info.size
                == plain_tar_info.size + 16 - plain_tar_info.size % 16 + 16 + 32
            )

    # Check the encrypted inner tars can be opened
    temp_decrypted = tmp_path.joinpath("decrypted")
    os.makedirs(temp_decrypted, exist_ok=True)
    with SecureTarArchive(main_tar_encrypted, password=password) as encrypted_archive:
        for inner_tar_file in inner_tar_files:
            encrypted_tar_info = encrypted_archive.tar.getmember(inner_tar_file)
            with encrypted_archive.extract_tar(encrypted_tar_info) as decrypted:
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
def test_encrypted_tar_inside_tar(
    tmp_path: Path, bufsize: int, enable_gzip: bool, inner_tar_files: tuple[str, ...]
) -> None:
    password = "hunter2"

    # Prepare test folder
    temp_orig = tmp_path.joinpath("orig")
    fixture_data = Path(__file__).parent.joinpath("fixtures/tar_data")
    shutil.copytree(fixture_data, temp_orig, symlinks=True)

    # Create an archive with encrypted inner tars
    main_tar = tmp_path.joinpath("backup.tar")
    with SecureTarArchive(
        main_tar, "w", bufsize=bufsize, password=password
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
                with outer_secure_tar_archive.extract_tar(tar_info) as decrypted:
                    with pytest.raises(
                        SecureTarReadError, match="The inner tar is not gzip or tar"
                    ):
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
            "securetar.SecureTarRootKeyContext._password_to_key",
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


def test_inner_tar_not_allowed_in_streaming(tmp_path: Path) -> None:
    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")

    outer_secure_tar_archive = SecureTarArchive(main_tar, "w", streaming=True)

    with pytest.raises(SecureTarError):
        with outer_secure_tar_archive:
            with outer_secure_tar_archive.create_tar("any.tgz", gzip=True):
                pass


def test_outer_tar_must_be_open(tmp_path: Path) -> None:
    # Create Tarfile
    main_tar = tmp_path.joinpath("backup.tar")
    outer_secure_tar_archive = SecureTarArchive(main_tar, "w")

    with pytest.raises(SecureTarError):
        with outer_secure_tar_archive.create_tar("any.tgz", gzip=True):
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

    outer_secure_tar_file = SecureTarFile(main_tar, "x", gzip=False)
    with pytest.raises(FileExistsError):
        outer_secure_tar_file.open()
