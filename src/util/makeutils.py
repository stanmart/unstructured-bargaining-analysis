import re


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
