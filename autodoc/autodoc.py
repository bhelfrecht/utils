#!/usr/bin/env python3

import os
import glob
import sys
import subprocess
import argparse
import datetime
import pkg_resources
import importlib
import operator
import xml.etree.ElementTree as ET
parser = argparse.ArgumentParser()
parser.add_argument('rootdir', type=str,
        help='Root directory for project')
parser.add_argument('-i', '--input', type=str, default=None,
        help='Input file for README generation')
parser.add_argument('-s', '--symlinks', action='store_true',
        help='Follow symbolic links')
parser.add_argument('-v', '--version', action='store_true',
        help='Get Python version information')
parser.add_argument('-x', '--excludehidden', action='store_true',
        help='Exclude files/directories starting with "."')
parser.add_argument('-n', '--nodoc', type=str, default=[], nargs='+',
        help='Names of directories to skip')
parser.add_argument('-u', '--unique', action='store_true',
        help='Write template sed file with unique file/directory names')
args = parser.parse_args()

# Even though the file and directory
# objects share many attributes and
# methods, use two separate classes
# to better distinguish them
# with isinstance, if needed

class FileObj(object):
    """
        Class for creating "file objects",
        which store the relevant file information
        for building the READMEs

        ---Attributes---
        ID: the assigned file ID, starting with 'F'
        path: the absolute file path
        description: a description of the file
        description_link: the copy link for the file description, 
            formatted as [ID to copy from from, replacement string]
    """
    def __init__(self, ID, path):
        self.ID = ID
        self.path = path
        self.description = '' 
        self.description_link = [None, None]

class DirObj(object):
    """
        Class for creating "directory objects",
        which store the relevant directory information
        for building the READMEs

        ---Attributes---
        ID: the assigned directory ID, starting with 'D'
        path: the absolute path to the directory
        description: a description of the directory
        description_link: the copy linki for the directory description,
            formatted as [ID to copy from, replacement string]
        notes: additional notes for the directory
        files: list of FileObjs for files in the directory
        all_dirs: list of DirObjs for folders in the directory,
            including subfolders
        dirs: list of DirObjs for folders in the directory,
            not including subfolders
    """
    def __init__(self, ID, path):
        self.ID = ID
        self.path = path
        self.description = ''
        self.description_link = [None, None]
        self.notes = ''
        self.files = []
        self.all_dirs = []
        self.dirs = []

def parse_replacestr(replacestr):
    """ 
        Parse the replacement string, e.g., "U,V;W,X;Y,Z",
        which replaces U with V, W with X, and Y with Z
        Return the parsed string as a list of pairs, e.g.,
        [[U, V], [W, X], [Y, Z]]

        ---Arguments---
        replacestr: the replacement string formatted as above
    """
    return [pair.split(',') for pair in replacestr.split(';')]

def sort_description_links(dictionary):
    """
        Sort the description links so that in the
        copying process we don't copy the description
        from a file that hasn't had its description
        filled yet

        ---Arguments---
        dictionary: dictionary of IDs; the key
            is the ID of the FileObj/DirObj to copy to,
            and the corresponding entry is the ID of the
            FileObj/DirObj to copy from
    """

    # Extract keys and sort them
    keys = list(dictionary.keys())
    sorted_keys = keys[:]
    items = []

    # Get key items
    for k in keys:
        items.append(dictionary[k])

    # Need ID in key before in 'items',
    # b/c otherwise we try to copy
    # a description that hasn't been
    # assigned yet
    for kdx, k in enumerate(keys):
        try:
            idx = items.index(k)
        except ValueError:
            pass
        else:
            if kdx > idx:
                p = sorted_keys.pop(kdx)
                sorted_keys.insert(idx, p)

    return sorted_keys

def copy_description(obj1, obj2):
    """
        Copy the description between FileObjs/DirObjs

        ---Arguments---
        obj1: FileObj/DirObj to copy description to
        obj2: FileObj/DirObj to copy description from
    """

    # The description to copy
    description_cpy = obj2.description
    
    # If we have a replacement string, do the replacement
    if obj1.description_link[1] is not None:
        for pair in obj1.description_link[1]:
            description_cpy = description_cpy.replace(pair[0], pair[1])
    
    # If the present description is empty, use the copied description
    if obj1.description == '':
        obj1.description = description_cpy

    # If the present description is not empty, append it to
    # the copied description
    else:
        obj1.description = '{:s}\n\t{:s}'.format(description_cpy,
                obj1.description)

def write_descriptions(fid, fd_list, write_time):

    # Write directory descriptions
    for obj in fd_list:
        dglob = sorted(glob.glob(obj.path))
        if len(dglob) > 1:

            # Write name 
            fid.write('### {:s}  \n'.format(os.path.split(obj.path)[1]))

            # For multiple directories in the glob, write out file path
            # and time of last modification for each, but just one description
            for globule in dglob:
                if write_time is True:
                    dmtime = os.path.getmtime(globule)
                    dmtime = datetime.datetime \
                            .utcfromtimestamp(dmtime).strftime('%Y-%m-%d %H:%M:%S')
                    fid.write('{:s} ({:s})  \n'.format(os.path.split(globule)[1], str(dmtime)))
                else:
                    fid.write('{:s}  \n'.format(os.path.split(globule)[1]))
            
            fid.write('\n')
            
        else:

            # Write directory name, modification time, and description
            if write_time is True:
                dmtime = os.path.getmtime(dglob[0])
                dmtime = datetime.datetime \
                            .utcfromtimestamp(dmtime).strftime('%Y-%m-%d %H:%M:%S')
                fid.write('### {:s} ({:s})  \n'.format(os.path.split(dglob[0])[1], str(dmtime)))
            else:
                fid.write('### {:s}  \n'.format(os.path.split(dglob[0])[1]))

        if obj.description != '':
            fid.write('{:s}  \n\n'.format(obj.description))
        else:
            fid.write('\n')

# Root working directory
args.rootdir = os.path.abspath(args.rootdir)

# Initialize dictionaries to hold the FileObjs and DirObjs;
# the dictionary keys are the FileObj/DirObj IDs, and the 
# corresponding entry is the FileObj/DirObj itself
dirs = {}
files = {}
dir_description_links = {}
file_description_links = {}

# An input file is supplied
if args.input is not None:

    # Parse the XML input file
    inputfile = ET.parse(args.input)
    project = inputfile.getroot()
    for directory in project:

        # Create object representing the directory
        dir_obj = DirObj(directory.attrib['id'], directory.attrib['path'])

        # Parse the tags for each directory
        for element in directory:

            # Get directory notes
            if element.tag == 'notes':
                dir_obj.notes = (element.text).strip()

            # Get directory description
            elif element.tag == 'description':
                dir_obj.description = (element.text).strip()

                # Get ID from which to copy the description
                if 'copy' in element.attrib:
                    dir_obj.description_link[0] = element.attrib['copy']
                    dir_description_links[dir_obj.ID] = dir_obj.description_link[0]

                # Parse the find-and-replace string for the description
                if 'replace' in element.attrib:
                    dir_obj.description_link[1] = \
                            parse_replacestr(element.attrib['replace'])

            # Get file descriptions
            elif element.tag == 'file':
                file_obj = FileObj(element.attrib['id'], 
                        os.path.join(directory.attrib['path'], element.attrib['name']))

                # Build dictionaries of options
                for description in element:

                    # Get the provided descriptions
                    file_obj.description = (description.text).strip()

                    # Get ID from which to copy the description
                    if 'copy' in description.attrib:
                        file_obj.description_link[0] = description.attrib['copy']
                        file_description_links[file_obj.ID] = file_obj.description_link[0]

                    # Parse the find-and-replace string for the description
                    if 'replace' in description.attrib:
                        file_obj.description_link[1] = \
                                parse_replacestr(description.attrib['replace'])

                # Build the dictionary of files
                # and append file objects to their parent directory
                dir_obj.files.append(file_obj)
                files[file_obj.ID] = file_obj

        # Build dictonary of directory objects
        dirs[dir_obj.ID] = dir_obj

    # Sort the order in which to access the dictionaries of files/directories
    # so that we don't copy a description that hasn't been assigned yet
    sorted_file_description_links = sort_description_links(file_description_links)
    sorted_dir_description_links = sort_description_links(dir_description_links)

    # Find child and grandchild directories for each DirObj
    # and sort alphabetically by path
    for d in dirs:
        dir_obj = dirs[d]

        # Look for directories with matching base paths
        for dd in dirs:
            test_obj = dirs[dd]

            # We have a directory directly within the directory of interest
            if os.path.split(test_obj.path)[0] == dir_obj.path:
                dir_obj.dirs.append(test_obj)
                dir_obj.all_dirs.append(test_obj)

            # We have a child or grandchild directory
            elif test_obj.path.startswith(dir_obj.path) and test_obj != dir_obj:
                dir_obj.all_dirs.append(test_obj)

        # Sort the lists of directories and files for
        # the directory object
        dir_obj.all_dirs.sort(key=operator.attrgetter('path'))
        dir_obj.dirs.sort(key=operator.attrgetter('path'))
        dir_obj.files.sort(key=operator.attrgetter('path'))

    # Build copied descriptions for files
    for sdl in sorted_file_description_links:
        file_obj = files[sdl]
        file_obj_match = files[file_obj.description_link[0]]
        copy_description(file_obj, file_obj_match)

    # Build recursively copied descriptions for directories
    for sdl in sorted_dir_description_links:
        dir_obj = dirs[sdl]
        dir_obj_match = dirs[dir_obj.description_link[0]]
        copy_description(dir_obj, dir_obj_match)

        # Match file descriptions in the directory
        for f, f_match in zip(dir_obj.files, dir_obj_match.files):
            f.description_link = [f_match.ID, dir_obj.description_link[1]]
            copy_description(f, f_match)

        # Match child directory descriptions
        for child, child_match in zip(dir_obj.all_dirs, dir_obj_match.all_dirs):
            child.description_link = [child_match.ID, 
                    dir_obj.description_link[1]]
            copy_description(child, child_match)

            # Match file descriptions in the child directories
            for f, f_match in zip(child.files, child_match.files):
                f.description_link = [f_match.ID, dir_obj.description_link[1]]
                copy_description(f, f_match)

    # Write the README files
    for d in dirs:
        dir_obj = dirs[d]
        g = open('{:s}/README.md'.format(dir_obj.path), 'w')

        # Get time of README generation (now)
        utctoday = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        g.write('Generated automatically on {:s}\n\n'.format(utctoday))

        # Write filepath
        g.write('# {:s}\n\n'.format(os.path.split(dir_obj.path)[1]))

        # Write descriptions for directories in this directory
        if len(dir_obj.dirs) > 0:
            g.write('## Directories\n')
            write_descriptions(g, dir_obj.dirs, write_time=False)

        # Write descriptions for files in this directory
        if len(dir_obj.files) > 0:
            g.write('## Files\n')
            write_descriptions(g, dir_obj.files, write_time=True)

        # Write notes
        if dir_obj.notes != '':
            g.write('## Notes\n{:s}\n'.format(dir_obj.notes))

else:

    # Template input file
    g = open('walk.xml', 'w')
    g.write('<project>\n')

    # Get files
    n_dirs = 0
    n_files = 0

    nodoc_paths = [os.path.abspath(nd) for nd in args.nodoc]
    unique_dirs = []
    unique_files = []

    for root, dirs, files, in os.walk(args.rootdir, 
            followlinks=args.symlinks):

        root_path = os.path.abspath(root)

        # Skip hidden files and directories
        if args.excludehidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')] 
            files[:] = [f for f in files if not f.startswith('.')] 

        # Skip requested directories
        dirs[:] = sorted([d for d in dirs if os.path.join(root_path, d) not in nodoc_paths])

        # Skip READMEs
        files[:] = sorted([f for f in files if not f.startswith('README')])

        # Write the template 'root' directories
        g.write("\t<dir id='D{:d}' path='{:s}'>\n".format(n_dirs, 
            root_path))
        g.write('\t\t<description>\n\t\t</description>\n')

        # Append unique directory names
        for d in dirs:
            if d not in unique_dirs:
                unique_dirs.append(d)

        # Write file name, date of last modification, and description
        # And append to unique file names
        for f in files:
            if f not in unique_files:
                unique_files.append(f)
            fid = 'F{:d}'.format(n_files)
            g.write("\t\t<file id='{:s}' name='{:s}'>\n".format(fid, f))
            g.write('\t\t\t<description>\n\t\t\t</description>\n')
            g.write("\t\t</file>\n")
            n_files += 1

        # Write placeholders for notes
        g.write('\t\t<notes>\n\t\t</notes>\n')
        g.write('\t</dir>\n')

        n_dirs += 1

    g.write('</project>')
    g.close()

    # Write template file for sed
    # Reference: https://stackoverflow.com/questions/16214828/sed-how-to-add-new-line-after-string-match-2-lines
    if args.unique:
        g = open('walk.sed', 'w')
        g.write('#!/usr/bin/sed -f\n\n')
        g.write('# Files\n')
        for uf in unique_files:
            g.write("/name='.*{:s}'/{{\nn\na\\\n}}\n\n".format(uf))

        g.write('\n')
        g.write('# Directories\n')
        for ud in unique_dirs:
            g.write("/path='.*{:s}'/{{\nn\na\\\n}}\n\n".format(ud))

        g.close()

if args.version:

    # Grep all the *.py and *.ipynb files
    s = subprocess.Popen(['grep', 'import', 
        '-r', 
        '--include=*.py', 
        '--include=*.ipynb',
        '--exclude-dir=.ipynb_checkpoints',
        '--exclude=autodoc.py',
        '{:s}'.format(args.rootdir)], 
        stdout=subprocess.PIPE, universal_newlines=True) # Decode (can use text=True for Py3.7+)

    # Read the stdout
    imports = s.communicate()[0]
    imports = imports.split('\n')
    imports.pop(-1) # This is an empty string (b/c of the split)

    # Parse the stdout to just retain module names by some stripping and splitting
    imports = [[i.replace('import ', '').replace(' as ', ' ').replace('from ', '') \
            for i in j.split(':')] for j in imports]
    imports = [i[1].replace('.', ' ').strip(' ",').rstrip('\\n').split()[0] for i in imports]
    modules = sorted(list(set(imports)))

    # Get version numbers and write information to file
    g = open('version_info.md', 'w')
    py_version = '.'.join([str(i) for i in sys.version_info])
    utctoday = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    g.write('Generated automatically on {:s}\n\n'.format(utctoday))
    g.write('# Python {:s}\n\n'.format(py_version))
    g.write('# Module Versions\n')
    all_module_names = []
    all_module_versions = []

    # Get all non-standard library module versions
    # https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules/32965521#32965521
    for sp in sys.path:
        for mv in pkg_resources.find_distributions(sp):
            all_module_names.append(mv.key)
            all_module_versions.append(mv.version)

    # Write module version, if found (could be a standard library module)
    for m in modules:
        if m in all_module_names:
            g.write('{:s}: {:s}\n'.format(m, all_module_versions[all_module_names.index(m)]))
        else:

            # Module and import name might mismatch, try direct import
            try:
                mm = importlib.import_module(m)
            except ImportError:

                # Probably we have a custom module 
                # not on the path, so we can't
                # assign a version number
                mm = None

            # Check a bunch of potential version attributes
            # b/c the version information of packages (if it exists)
            # isn't always in module.__version__
            if hasattr(mm, '__version__'):
                v = mm.__version__
            elif hasattr(mm, 'version'):
                v = mm.version
            elif hasattr(mm, 'Version'):
                v = mm.Version
            elif hasattr(mm, 'VERSION'):
                v = mm.VERSION
            elif hasattr(mm, 'version_info'):
                v = mm.version_info
            else:
                v = 'version not found'

            g.write('{:s}: {:s}\n'.format(m, v))

    g.write('\n')

    # Get info on any Github repositories in the project
    g.write('# Github Repositories\n')
    for root, dirs, files in os.walk(args.rootdir):
        for d in dirs:
            
            # Find a git repository
            if d == '.git':

                # Parse the current branch
                r = open('{:s}/.git/HEAD'.format(root), 'r')
                ref = r.readline().strip().split()[1]
                branch = ref.split('/')[-1]
                r.close()

                # Get the current commit of the current branch
                c = open('{:s}/.git/{:s}'.format(root, ref), 'r')
                commit = c.readline().strip()
                c.close()

                # Write the repository information
                g.write('{:s}:\n'.format(root))
                g.write('\tBranch {:s}, Commit {:s}\n'.format(branch, commit))

    g.write('\n')
    g.close()
