"""MeesaJarJar text node - extracted from JarJarSaveText.py (Old pack)."""

import os


class JarJarSaveText:
    """Save text_content STRING to filepath on disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filepath": ("STRING", {
                    "default": "/tmp/output.txt",
                    "multiline": False,
                }),
                "text_content": ("STRING", {
                    "default": "Hello, JarJar!",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "JarJar/Text"
    DESCRIPTION = "Writes text_content to filepath (creates dirs). Returns status STRING."

    def save_text(self, filepath, text_content):
        try:
            filepath = filepath.strip()
            text_content = text_content.strip()

            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)

            status = f"Successfully saved text to {filepath}"
            print(status)
            return (status,)

        except OSError as e:
            error_msg = f"Error saving file: {str(e)}"
            print(error_msg)
            return (error_msg,)


NODE_CLASS_MAPPINGS = {
    "JarJarSaveText": JarJarSaveText
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JarJarSaveText": "JarJar Save Text"
}
