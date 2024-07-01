import pathlib

out_dir = pathlib.Path("out/presentation")
src_dir = out_dir / "src"
output_files = src_dir.glob("presentation/*.html")
for output_file in output_files:
    output_file.rename(out_dir / output_file.name)
(src_dir / "presentation").rmdir()
src_dir.rmdir()
