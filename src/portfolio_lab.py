"""
This is the main script file for portfolio lab.

From the documentation, a DPG script must:
  - Create the context    |  `create_context()`
  - Create the viewport   |  `create_viewport()`
  - Setup dearpygui       |  `setup_dearpygui()`
  - Show the viewport     |  `show_viewport()`
  - Start dearpygui       |  `start_dearpygui()`
  - Clean up the context  |  `destroy_context()`
"""


import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import tkinter as tk



# |--------------------------------- Dev Toggles ----------------------------------|
# Used for dev, should be False in releases
SHOW_DEMO = True

# Card padding constants
CARD_OUT_X_PAD = 0.05   # 5% padding L / R
CARD_OUT_Y_PAD = 0.06   # 6% padding top / bottom
CARD_GAP_X     = 0.03   # Between card padding L / R
CARD_GAP_Y     = 0.04   # Between card padding top / bottom

CARD_ROUNDING    = 0.06   # rounding relative to card size
CARD_INTERNAL_PAD = 0.06   # padding inside card content
# |--------------------------------- Dev Toggles ----------------------------------|


# |----------------------- Screen Initialization Utilities ------------------------|
def compute_layout(w, h):
    outer_pad_x = int(w * CARD_OUT_X_PAD)
    outer_pad_y = int(h * CARD_OUT_Y_PAD)
    gap_x       = int(w * CARD_GAP_X)
    gap_y       = int(h * CARD_GAP_Y)

    # Card width from 3 columns
    card_w = (w - 2 * outer_pad_x - 2 * gap_x) / 3.0

    # Card height from 2 rows
    card_h = (h - 2 * outer_pad_y - 1 * gap_y) / 2.0

    # Enforce card slightly wider than tall
    card_h = min(card_h, card_w * 0.85)

    card_w = int(card_w)
    card_h = int(card_h)

    rounding = int(min(card_w, card_h) * CARD_ROUNDING)
    inner_pad_y = int(card_h * CARD_INTERNAL_PAD)

    return outer_pad_x, outer_pad_y, gap_x, gap_y, card_w, card_h, rounding, inner_pad_y


def draw_card(card_w, card_h, rounding, inner_pad_y, label):
    # Background (rounded rect)
    with dpg.drawlist(width=card_w, height=card_h):
        dpg.draw_rectangle(
            (0, 0),
            (card_w, card_h),
            color=(70, 70, 70, 255),
            fill=(35, 35, 35, 255),
            rounding=rounding,
            thickness=1,
        )

    # Foreground content area (transparent container on top)
    with dpg.child_window(width=card_w, height=card_h, border=False):
        dpg.add_spacer(height=inner_pad_y)
        dpg.add_text(label)
        dpg.add_text("Widgets go here")


def init_windows(w, h):
  """
  Inputs:
    - w : total screen width
    - h: total screen height
  """

  # Portfolio window takes up 2/3 of the screen
  with dpg.window(
      tag="Portfolio Window",
      pos=(0, 0),
      width=int(w * (2/3)),
      height=h,
      no_move=True,
      no_title_bar=True
    ):
    # Any dpg window code in this block will apply to the above window.
    #   - We can later modify it by specifying parent="Portfolio Window"

    # The 'screen' width/height here references this windows resolution, not the global resolution
    screen_width  = int(w * (2 / 3))
    screen_height = h
    outer_pad_x, outer_pad_y, gap_x, gap_y, card_w, card_h, rounding, inner_pad_y = compute_layout(
        screen_width, screen_height
    )

    # Outer padding
    dpg.add_spacer(height=outer_pad_y)

    # Row 1 (3 cards), with left padding via spacer
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=outer_pad_x)

        for i in range(3):
            with dpg.group():
                draw_card(card_w, card_h, rounding, inner_pad_y, f"Cell {i+1}")
            if i < 2:
                dpg.add_spacer(width=gap_x)

        dpg.add_spacer(width=outer_pad_x)

    # Row gap
    dpg.add_spacer(height=gap_y)

    # Row 2 (2 cards)
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=outer_pad_x)

        for i in range(2):
            with dpg.group():
                draw_card(card_w, card_h, rounding, inner_pad_y, f"Cell {i+4}")
            if i < 1:
                dpg.add_spacer(width=gap_x)

        # (Optional) Keep the third slot empty but aligned visually:
        # add a spacer equal to a missing card + one gap
        # dpg.add_spacer(width=card_w + gap_x)

        # dpg.add_spacer(width=outer_pad_x)

# |----------------------- Screen Initialization Utilities ------------------------|


# |---------------------------- Screen Initialization -----------------------------|
# We load tkinter literally just to fetch the screen specs
def screen_init():
  root = tk.Tk()
  root.withdraw()
  screen_width = root.winfo_screenwidth()
  screen_height = root.winfo_screenheight()
  root.destroy()

  dpg.create_context()
  dpg.create_viewport(title='Portfolio Lab', width=screen_width, height=screen_height)

  # |---------------------------- DEFINE UI TREE HERE ----------------------------|
  init_windows(screen_width, screen_height)
  # |---------------------------- DEFINE UI TREE HERE ----------------------------|

  # Anything defined after these calls simply won't appear
  dpg.setup_dearpygui()
  dpg.show_viewport()
# |---------------------------- Screen Initialization -----------------------------|


# |---------------------------------- Close App -----------------------------------|
def close():
  dpg.destroy_context()
# |---------------------------------- Close App -----------------------------------|


# |-------------------------------- Main Function ---------------------------------|
def main():
  screen_init()

  if SHOW_DEMO:
    # Demo only shows up *inside* of the viewport we defined for the main window
    demo.show_demo()

  dpg.start_dearpygui()

  """
  TODO:
  (1) - Draw UI That looks like diagram_UI.png
  (2) - Integrate Markowitz Modeling into UI
  """

#  while(True):
#    print("What?")

  close()


if __name__ == "__main__":
  main()
