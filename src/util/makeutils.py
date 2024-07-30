import json
import re
import subprocess

import requests
import typer

app = typer.Typer()


def find_quarto_images(quarto_file: str, remove_leading_slash=True):
    with open(quarto_file, "r") as file:
        quarto_contents = file.read()

    patterns = [
        re.compile(r"\!\[.*?\]\((.*?)\)"),
    ]

    images = sum((pattern.findall(quarto_contents) for pattern in patterns), [])

    if remove_leading_slash:
        images = [section.lstrip("/") for section in images]

    return images


def find_opened_files(quarto_file: str, remove_leading_slash=True):
    with open(quarto_file, "r") as file:
        quarto_contents = file.read()

    patterns = [
        re.compile(r"open\(\"(.*?)\",.*?\)"),
    ]

    images = sum((pattern.findall(quarto_contents) for pattern in patterns), [])

    if remove_leading_slash:
        images = [section.lstrip("/") for section in images]

    return images


@app.command()
def find_input_files(
    tex_file: str,
    remove_prefix: str = "",
    print_console: bool = False,
):
    """Find all of the files that are included, inputted or includegraphics'
    in a tex file.
    Args:
        tex_file: The path of the tex file.
        add_tex_folder: Whether to add the folder of the tex file to the
            paths of the included files.
        print_console: Whether to print the paths of the included files
    Returns:
        List[str]: the list of included files
    """

    with open(tex_file, "r") as file:
        tex_contents = file.read()

    patterns = [
        re.compile(r"\\input\{(.*)\}"),
        re.compile(r"\\include\{(.*)\}"),
        re.compile(r"\\includegraphics(?:\[.*\])?\{(.*)\}"),
        re.compile(r"\\pgfplotstableread(?:\[.*\])\{(.*)\}"),
    ]

    paths = sum((pattern.findall(tex_contents) for pattern in patterns), [])

    if remove_prefix:
        paths = [path.lstrip(remove_prefix) for path in paths]

    if print_console:
        for path in paths:
            print(path)

    return paths


@app.command()
def collect_latex_packages(
    dep_files: list[str],
    output_file: str = "tl_packages.txt",
    add_biber: bool = False,
    add_latexmk: bool = False,
    check_against_tl: bool = False,
    force_add: list[str] = [],
    print_console: bool = False,
) -> list[str]:
    """Collect all of the latex packages from the dependency files.
    Args:
        dep_files: The list of dependency files.
        output_file: The path of the output file.
    Returns:
        List[str]: the list of latex packages
    """

    deps = set()

    for dep_file in dep_files:
        with open(dep_file, "r") as file:
            dep_contents = file.read()

        pattern = re.compile(r"\*\{package\}\{(.*?)\}")

        deps.update(pattern.findall(dep_contents))

    if add_biber:
        deps.add("biber")

    if add_latexmk:
        deps.add("latexmk")

    packages = set()
    for dep in deps:
        req = requests.get(f"https://www.ctan.org/json/2.0/pkg/{dep}")
        if req.status_code == 200:
            package_data = json.loads(req.text)
            if "texlive" in package_data:
                packages.add(package_data["texlive"])
            else:
                print(f"No texlive package found for package {dep}")
        else:
            print(f"Package {dep} not found on CTAN")

    if check_against_tl:
        tl_packages = set(
            subprocess.getoutput("tlmgr --verify-repo=none info --data=name").split(
                "\n"
            )
        )
        packages &= tl_packages

    packages |= set(force_add)

    package_list = sorted(list(packages))

    if print_console:
        print(f"Found {len(package_list)} dependencies:")
        for package in package_list:
            print(f" > {package}")

    if output_file:
        with open(output_file, "w") as file:
            for package in package_list:
                file.write(package + "\n")

    return package_list


if __name__ == "__main__":
    app()
