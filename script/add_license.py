#!/usr/bin/env python3


import glob
import os
import re


def mit_license_template():
    return """MIT License

Copyright (c) {year} {copyright_holders}

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""


def mit_license(year: str = "2024--present", copyright_holders: str = "ByteDance Ltd. and/or its affiliates"):
    return mit_license_template().format(year=year, copyright_holders=copyright_holders)


def extra_paragraphs():
    return """The license for this source file is stated above. This paragraph and
the following one are not part of the license statement.

The file doc/CREDITS.txt lists serval external projects and their
licenses, which served as valuable guidance for this project. Although
not all of these projects are directly referenced in every source
file, this source file complies with the licenses of all listed
projects."""


def replace_paragraph(content: str, start: str, end: str, replace: str):
    pattern = rf"({re.escape(start)}.*?{re.escape(end)})"
    match = re.search(pattern, content, flags=re.DOTALL)
    if match:
        return re.sub(pattern, replace, content, count=1, flags=re.DOTALL)
    else:
        return "\n".join([replace, content, ""])


def to_cxx(sections: list[str]):
    line0 = "/* -------------------------------------------------------------------------- *"
    lineD = " * -------------------------------------------------------------------------- *"
    lineE = " *                                                                            *"
    line1 = " * -------------------------------------------------------------------------- */"

    new_lines = []
    new_lines.append(line0)
    for idx, s in enumerate(sections):
        if idx > 0:
            new_lines.append(lineE)
            new_lines.append(lineD)
            new_lines.append(lineE)
        lines = s.split("\n")
        assert len(line1) - len(line0) == 1
        for l in lines:
            l2 = l.strip()
            padding = len(line0) - 3 - len(l2) - 2
            assert padding >= 0
            new_lines.append(" * " + l2 + " " * padding + " *")
    new_lines.append(line1)
    return "\n".join(new_lines), line0, line1


def current_dir():
    return os.path.dirname(__file__)


def get_files1():
    files = []
    files.extend(glob.glob(current_dir() + "/../clbk/**/*.h", recursive=True))
    files.extend(glob.glob(current_dir() + "/../pybind11/**/*.cpp", recursive=True))
    return files


def get_files2():
    files = []
    files.extend(glob.glob(current_dir() + "/../OpenMMPlugin/**/*.h", recursive=True))
    files.extend(glob.glob(current_dir() + "/../OpenMMPlugin/**/*.cpp", recursive=True))
    return files


def mainfunc():
    license_str = mit_license()
    extra_str = extra_paragraphs()
    files1 = get_files1()
    files2 = get_files2()

    file0 = current_dir() + "/../LICENSE"
    with open(file0, "w") as fw:
        fw.write(license_str + "\n")

    for f in files1:
        with open(f) as fr:
            content = fr.read()
            l, l0, l1 = to_cxx([license_str])
            new_content = replace_paragraph(content, l0, l1, l)
        with open(f, "w") as fw:
            fw.write(new_content)

    for f in files2:
        with open(f) as fr:
            content = fr.read()
            l, l0, l1 = to_cxx([license_str, extra_str])
            new_content = replace_paragraph(content, l0, l1, l)
        with open(f, "w") as fw:
            fw.write(new_content)


if __name__ == "__main__":
    mainfunc()
