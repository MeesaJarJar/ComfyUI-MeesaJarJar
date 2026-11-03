"""MeesaJarJar list/util nodes - extracted from Old pack:
- JarJarCustom.py (ListFilesNode)
- JarJarList.py (JarJarTextList, JarJarListItem)
- jarjar_range_int_large.py (JarJarRangeIntLarge, fixed mapping key)
"""

import csv
import os


class ListFilesNode:
    """List files in a folder, optionally stripping extensions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"multiline": False, "default": "./", "lazy": True}),
                "include_extensions": (["enable", "disable"],)
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("file_list",)
    FUNCTION = "list_files"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Lists files in path folder. disable strips extensions. Returns LIST."

    def list_files(self, path, include_extensions):
        try:
            files = os.listdir(path)
            files = [f for f in files if os.path.isfile(os.path.join(path, f))]
            if include_extensions == "disable":
                files = [os.path.splitext(f)[0] for f in files]
        except OSError as e:
            print(f"Error listing files: {e}")
            files = []
        return (files,)


class JarJarFolderFilesByExt:
    """List top-level files in folder filtered by extension, returns full paths."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"multiline": False, "default": "./", "lazy": True}),
                "extension": ("STRING", {"multiline": False, "default": ".ply"}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "list_files_by_ext"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Lists top-level files only (no subfolders), filters by extension (.ply or .ply,.glb or ply), returns sorted full paths as LIST."

    def list_files_by_ext(self, folder, extension):
        folder = (folder or "").strip().strip("\"'").strip()
        if not folder or not os.path.isdir(folder):
            print(f"JarJarFolderFilesByExt: folder not found: {folder}")
            return ([],)
        ext_set = set()
        for part in (extension or "").replace(";", ",").split(","):
            for tok in part.split():
                t = tok.strip().lower()
                if not t:
                    continue
                if not t.startswith("."):
                    t = "." + t
                if t == ".":
                    continue
                ext_set.add(t)
        if not ext_set:
            return ([],)
        try:
            names = os.listdir(folder)
        except OSError as e:
            print(f"JarJarFolderFilesByExt error listing: {e}")
            return ([],)
        paths = []
        for n in names:
            fp = os.path.join(folder, n)
            if not os.path.isfile(fp):
                continue
            if os.path.splitext(n)[1].lower() in ext_set:
                paths.append(fp)
        paths.sort()
        return (paths,)


class JarJarTextList:
    """Split newline-delimited text block into a LIST (one line per item)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_block": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "First line\nSecond line\nThird line"
                    }
                ),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("lines",)
    FUNCTION = "split_lines"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Splits text_block on newlines, trims, drops empties. Returns LIST."

    def split_lines(self, text_block):
        lines = [line.strip() for line in text_block.splitlines() if line.strip()]
        return (lines,)


class JarJarListItem:
    """Get LIST item by index (safe, returns empty string if out of range)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_in": ("LIST", {}),
                "index": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("item",)
    FUNCTION = "get_item"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Returns list_in[index] as STRING, or empty string if out of range."

    def get_item(self, list_in, index):
        try:
            return (list_in[index],)
        except IndexError:
            return ("",)


class JarJarCSVFileToList:
    """Load a text/CSV file from disk into a LIST (one entry per row)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"multiline": False, "default": "./list.csv", "lazy": True}),
                "delimiter": ("STRING", {"multiline": False, "default": ","}),
                "column": ("INT", {"default": 0, "min": 0, "step": 1}),
                "has_header": ("BOOLEAN", {"default": False}),
                "skip_empty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("items",)
    FUNCTION = "load_csv_list"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Reads file_path (.csv/.txt) and returns one LIST entry per row (column index). Plug into JarJar List Item."

    @classmethod
    def IS_CHANGED(cls, file_path, delimiter=",", column=0, has_header=False, skip_empty=True):
        try:
            return (os.path.getmtime(file_path.strip()),)
        except OSError:
            return (None,)

    def load_csv_list(self, file_path, delimiter=",", column=0, has_header=False, skip_empty=True):
        items = []
        try:
            file_path = (file_path or "").strip()
            if not file_path or not os.path.isfile(file_path):
                print(f"JarJarCSVFileToList: file not found: {file_path}")
                return ([],)
            if delimiter == "\\t":
                delimiter = "\t"
            if not delimiter:
                delimiter = ","
            column = max(0, int(column))
            with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if i == 0 and has_header:
                        continue
                    if not row:
                        continue
                    value = row[column].strip() if column < len(row) else ""
                    if skip_empty and not value:
                        continue
                    items.append(value)
        except (OSError, ValueError, csv.Error) as e:
            print(f"JarJarCSVFileToList error: {e}")
            return ([],)
        return (items,)


class JarJarRangeIntLarge:
    """Inclusive INT range -> LIST."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0}),
                "end": ("INT", {"default": 10}),
                "step": ("INT", {"default": 1}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("ints",)
    FUNCTION = "range_int_large"
    CATEGORY = "JarJar/Utils"
    DESCRIPTION = "Inclusive range(start, end, step) as LIST. Caution: huge ranges can OOM."

    def range_int_large(self, start, end, step):
        step = int(step)
        if step == 0:
            raise ValueError("step must not be zero")

        values = list(range(int(start), int(end) + (1 if step > 0 else -1), step))
        return (values,)


NODE_CLASS_MAPPINGS = {
    "ListFilesNode": ListFilesNode,
    "JarJarFolderFilesByExt": JarJarFolderFilesByExt,
    "JarJarTextList": JarJarTextList,
    "JarJarCSVFileToList": JarJarCSVFileToList,
    "JarJarListItem": JarJarListItem,
    "JarJarRangeIntLarge": JarJarRangeIntLarge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ListFilesNode": "List Files Node",
    "JarJarFolderFilesByExt": "JarJar: Folder Files by Extension",
    "JarJarTextList": "JarJar Text List",
    "JarJarCSVFileToList": "JarJar CSV File To List",
    "JarJarListItem": "JarJar List Item",
    "JarJarRangeIntLarge": "JarJar Range Int Large",
}
