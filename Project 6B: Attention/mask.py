import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    for i, token_id in enumerate(inputs.input_ids[0]):
        if token_id == mask_token_id:
            return i
    return None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    rgb = round(255 * attention_score.numpy())
    return (rgb, rgb, rgb)

def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    for i, attention in enumerate(attentions):
        for j in range(len(attention[0])):
            generate_diagram(
                i + 1,
                j + 1,
                tokens,
                attentions[i][0][j]
            )


def generate_diagram(layer_idx, head_idx, tokens, attention):
    """
    Create and save an attention heatmap for a specific layer/head.
    Rows and columns correspond to tokens and cell brightness reflects
    the attention score.
    """

    token_count = len(tokens)
    canvas_len = PIXELS_PER_WORD + GRID_SIZE * token_count

    # base canvas
    canvas = Image.new("RGBA", (canvas_len, canvas_len), "black")
    painter = ImageDraw.Draw(canvas)

    # ----- draw token labels -----
    for pos, word in enumerate(tokens):

        offset = PIXELS_PER_WORD + pos * GRID_SIZE

        # draw row label (left side)
        left, top, right, bottom = painter.textbbox((0, 0), word, font=FONT)
        text_width = right - left

        painter.text(
            (PIXELS_PER_WORD - text_width, offset),
            word,
            fill="white",
            font=FONT
        )

        # draw column label (rotated)
        temp_layer = Image.new("RGBA", (canvas_len, canvas_len), (0, 0, 0, 0))
        temp_draw = ImageDraw.Draw(temp_layer)

        temp_draw.text(
            (canvas_len - PIXELS_PER_WORD, offset),
            word,
            fill="white",
            font=FONT
        )

        rotated = temp_layer.rotate(90)
        canvas.paste(rotated, mask=rotated)

    # ----- draw attention grid -----
    for row_index, row_scores in enumerate(attention):
        y0 = PIXELS_PER_WORD + row_index * GRID_SIZE

        for col_index, score in enumerate(row_scores):
            x0 = PIXELS_PER_WORD + col_index * GRID_SIZE
            shade = get_color_for_attention_score(score)

            painter.rectangle(
                (x0, y0, x0 + GRID_SIZE, y0 + GRID_SIZE),
                fill=shade
            )

    # ----- save output -----
    filename = f"Attention_Layer{layer_idx}_Head{head_idx}.png"
    canvas.save(filename)


if __name__ == "__main__":
    main()
