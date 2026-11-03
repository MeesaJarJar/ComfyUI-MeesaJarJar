class JarJarIntToString:
    """Convert an INT to STRING / TEXT."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0,
                    "min": -2147483648,
                    "max": 2147483647,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("STRING", "TEXT")
    FUNCTION = "convert"
    CATEGORY = "JarJar/Utils"
    OUTPUT_NODE = False

    def convert(self, value: int):
        text = str(value)
        return (text, text)
