from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT = Path("assets")
OUTPUT.mkdir(parents=True, exist_ok=True)

SIZE = 256
image = Image.new("RGBA", (SIZE, SIZE), (17, 19, 24, 255))
draw = ImageDraw.Draw(image)

draw.rounded_rectangle(
    (14, 14, SIZE - 14, SIZE - 14),
    radius=52,
    fill=(50, 94, 230, 255),
)

try:
    font = ImageFont.truetype("arial.ttf", 128)
except OSError:
    font = ImageFont.load_default()

text = "W"
box = draw.textbbox((0, 0), text, font=font)
x = (SIZE - (box[2] - box[0])) / 2
y = (SIZE - (box[3] - box[1])) / 2 - box[1]

draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
image.save(
    OUTPUT / "webflyx.ico",
    sizes=[
        (16, 16),
        (32, 32),
        (48, 48),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)

print("Created assets/webflyx.ico")
