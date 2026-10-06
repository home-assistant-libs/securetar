# Secure Tar

Secure Tarfile library

SecureTar is a streaming wrapper around Python's `tarfile` module. It
creates and reads tar archives that contain inner tar files, which can be
encrypted. It is the archive format used for Home Assistant backups.

## Archive layout

A SecureTar archive is a plain, uncompressed outer tar in PAX format. Each
member of the outer tar is an inner tar file, optionally gzip compressed and
optionally encrypted. Metadata files such as `backup.json` can be added to
the outer tar directly.

Encrypted inner tar files start with a SecureTar header followed by the
ciphertext. Two header versions can be written:

| Version | Cipher                          | Key derivation                     |
|---------|---------------------------------|------------------------------------|
| 2       | AES-128-CBC                     | 100 rounds of SHA-256 (legacy)     |
| 3       | XChaCha20-Poly1305 secretstream | Argon2id, per-file BLAKE2b subkeys |

Version 3 is the default. It authenticates the data, detects truncation, and
allows the password to be checked without decrypting the whole file. Version 1
(AES-128-CBC without a header) can still be read.

## Creating an archive

```python
from pathlib import Path

from securetar import SecureTarArchive, atomic_contents_add

with SecureTarArchive(Path("backup.tar"), "w", password="hunter2") as archive:
    with archive.create_tar("./homeassistant.tar.gz", gzip=True) as inner_tar:
        atomic_contents_add(
            inner_tar,
            Path("/config"),
            file_filter=lambda path: path.name == "home-assistant.log",
            arcname=".",
        )

    with archive.create_tar("./share.tar.gz", gzip=True) as inner_tar:
        atomic_contents_add(
            inner_tar,
            Path("/share"),
            file_filter=lambda _: False,
            arcname=".",
        )
```

`create_tar` returns a context manager that yields a regular
`tarfile.TarFile`. Everything written to it is streamed into the outer tar;
no temporary files are needed. The inner tar is encrypted when the archive was
opened with a `password` or a `root_key_context`. Leave both out to create a
plain archive.

`atomic_contents_add` recursively adds a directory. The `file_filter`
callback receives the archive-relative `PurePath` of each item and returns
`True` to exclude it. Errors while adding a file raise `AddFileError`, which
carries the offending path.

Pass `create_version=2` to `SecureTarArchive` to write the legacy AES format.

## Reading an archive

Iterate the outer tar to find the inner tar files, then use `extract_tar` to
get a decrypted stream, or wrap a member in `SecureTarFile` to read the inner
tar directly:

```python
from pathlib import Path

from securetar import SecureTarArchive, SecureTarFile

with SecureTarArchive(Path("backup.tar"), "r", password="hunter2") as archive:
    for member in archive.tar:
        if not member.name.endswith(".tar.gz"):
            continue

        # Decrypted bytes of the inner tar
        with archive.extract_tar(member) as decrypted:
            data = decrypted.read(1024 * 1024)

        # Or open the inner tar and extract its contents
        fileobj = archive.tar.extractfile(member)
        with SecureTarFile(fileobj=fileobj, password="hunter2") as inner_tar:
            inner_tar.extractall(Path("/restore"), filter="data")
```

Encrypted inner tar files are opened in stream mode because the ciphertext is
not seekable. Random access to members, such as `getmember` followed by
`extractfile`, is therefore not possible; read members sequentially.

A wrong password raises `InvalidPasswordError` for version 3 files. For
version 1 and 2 files, which carry no key check, a wrong password raises
`SecureTarReadError` once the decrypted data turns out not to be a tar.

## Validating

`SecureTarArchive.validate_password(member)` checks the password against an
encrypted inner tar without reading all of it.
`SecureTarArchive.validate(member)` additionally decrypts the whole member,
which for version 3 verifies integrity and detects truncation. Both consume
the member's stream. `SecureTarFile` offers the same two methods for a single
inner tar.

## Re-encrypting existing tar files

`import_tar` encrypts an existing plaintext tar and adds it to the archive
without writing to disk. The source `TarInfo` must have `size` set.

```python
with (
    SecureTarArchive(Path("plain.tar"), "r") as source,
    SecureTarArchive(Path("encrypted.tar"), "w", password="hunter2") as target,
):
    for member in source.tar:
        target.import_tar(source.tar.extractfile(member), member)
```

`get_archive_max_ciphertext_size(plaintext_size, version, number_of_inner_tar_files)`
returns an upper bound for the encrypted archive size, useful when the size
has to be announced before the data is streamed.

## Sharing key derivation between archives

Deriving the version 3 root key with Argon2id is deliberately slow. A
`SecureTarRootKeyContext` performs that derivation once and can be passed to
several `SecureTarArchive` instances instead of a password.

Passing the same `derived_key_id` to `create_tar` or `import_tar` reuses the
per-file key and nonce, so importing the same plaintext tar into two archives
produces byte-identical ciphertext. Use this only when identical output is
required, for example to upload one backup to several locations; otherwise
leave `derived_key_id` unset so every inner tar gets a fresh key.

```python
from securetar import SecureTarArchive, SecureTarRootKeyContext

root_key_context = SecureTarRootKeyContext("hunter2")

with SecureTarArchive(Path("plain.tar"), "r") as source:
    for name in ("backup1.tar", "backup2.tar"):
        with SecureTarArchive(Path(name), "w", root_key_context=root_key_context) as target:
            for member in source.tar:
                target.import_tar(
                    source.tar.extractfile(member), member, derived_key_id=member.name
                )
```
