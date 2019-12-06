# Autodoc
A tool to (hopefully) help make data management
a little less time-consuming

## autodoc.py

**Author**  
Benjamin A. Helfrecht

**Description**  
A python script that serves as an assistant for generating
metadata for a project. Currently the tool supports automatic
generation of time-of-last-modification for files and
manual descriptions of files and directories

**Usage**  
`autodoc.py ROOTDIR [-h] [-i INPUT] [-s] [-v] [-x] [-n NODOC [NODOC ...]] [-u]`

-h, --help; Display the help message

-i, --input  INPUT;  use an input XML file (format explained later) to generate 
README.md files for the project

-s, --symlinks; Follow symlinks in generating the template XML file

-v, --version; Write an output file with version information of Python modules
and commit hash of Github directories

-x, --excludehidden; Exclude files and directories starting with "." in file search
to build the template XML file

-n NODOC [NODOC ...], --nodoc NODOC [NODOC ...]; Names of directories to skip 
(i.e., do not add the contents of these directories to the template XML file)

-u, --unique; Create a template sed file for unique file and directory names;
useful when descriptions need to be duplicated for files with the same name

**Instructions**  
1.  First we wish to generate a template XML and sed file for the project.
    To do this, we first run `autodoc.py ROOTDIR -x -u`. 
    Omit the `-x` option if you need to documuent hidden files/directories
    and the `-u` option if you have no need to generate duplicate descriptions
    for similarly named files. Add the `-s` option if you want to document files
    accessed through a symlink. If there are directories, say `ROOTDIR/D1` and `ROOTDIR/D2`
    whose contents we wish not to document, pass also `-n ROOTDIR/D1 ROOTDIR/D2`.

2.  Take a look at the XML file (`walk.xml`). Each directory is contained in a `dir` tag, 
    with subtags under which a description or notes can be written. 
    Furthermore, there exists a `file` subtag for each file contained in the directory, 
    with its own description subtag.
    Each file and directory is given a unique ID attribute, which may be useful later.\
    Directories also have the `path` attribute, while files have the `name` attribute.
    In other words, the absolute file path for a file is the concatenation of the `path`
    from the parent directory and the `name` of the file.

3.  Now the task is to fill in the template XML file descriptions to document the files.
    However, for large projects, this can be impossible (the template XML can easily
    run to tens of thousands of lines). In such cases, if each file/directory has to have
    a totally unique description, you're out of luck. However, if many descriptions will be
    duplicated, or duplicated while only changing a few words or numbers, this tool should
    (hopefully) help. There are a number of options for duplicating descriptions:

    1.  Use the template `sed` file (`walk.sed`). This is recommended for most cases.
        To fill in descriptions, just write the description between the `a\` and `}`
        lines. Running the sed file afterwards (`sed -f walk.sed walk.xml > in.xml`) will
        populate the description tags of the XML file. For descriptions spanning
        more than one line (i.e., by pressing ENTER), a backslash "\" is required at the end
        of the line in the sed file.
        
        If we have descriptions we want to duplicate (for example, having a common description
        for `a1.txt`, `a2.txt`, `a3.txt`, we replace one of the file tags in the
        template XML with `a*.txt` (standard file globbing notation), and remove the others.
        This is accomplished in the sed file with the lines `s/a1\.txt/a*.txt/` and 
        `/a[1-9]\.txt/{N;N;N;d}`. The latter can alternatively be written as
        `/a[^\*]\.txt/{N;N;N;d}`. Then change the line in the sed file
        `/name='.*a1\.txt'/{` to `/name='.*a\*\.txt'/{` and write the description below.
        You can delete the code blocks for the other similar files, or delete them.
        With the globbed file description, we can write something like: 
        "a{X}.txt is the Xth file of a" so that one description suffices for all the `a*.txt` files.

    2.  To do a more fine-tuned copying, we can use the `copy` and `replace` attibutes of the
        file and directory description tags. To copy a file description for say, change
        the description tag line to, for example, 
        `<description copy='F2' replace='apple,banana;red,blue'>.
        This will copy the description from the file with the ID F2 and replace all substrings
        of "apple" with "banana" and all substrings of "red" with "blue".
        This is useful in cases where we want to duplicate a description but change just
        a few words or numbers. Using these options we can also chain duplications, e.g.,
        have file F3 copy from F4, and have F4 copy from F1. Just be careful not to make
        an infinite loop. Similarly, we can copy descriptions for directories in the same way,
        but the mechanics are a bit different, so use with discretion 
        (generally the sed file is more helpful in these situations anyway).
        In contrast to copying file descriptions, copying directory descriptions directly copies
        the description of the directory in addition to all descriptions
        of child files and directories. In other words, only use this option to copy
        descriptions from directories with identically named contents.
        Identically named contents are assumed in the implementation, so one additional file
        could screw up the entire documentation. Use this capability with care. 
        Moreover, for doing this with globbed files, make sure that all of the globs
        are consistent between the source and destination directories, e.g.,
        don't copy from a directory that globs `a*.txt` to a directory that still lists
        `a1.txt`, `a2.txt`, and `a3.txt`. Both directories should use `a*.txt` in the XML.
        I will probably remove this capability in the future make the behavior the same
        as file copying. Copying identical contents of directories is also simpler and more
        straightforward with the template sed file.

4.  Generate the README files and version information, if desired. We do this using the now-complete
    XML file and the `-i` option of `autodoc.py`. Run `autodoc.py ROOTDIR -i in.xml -v`.
    Omit the `-v` option to skip generating version information.
