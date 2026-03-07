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

from pathlib import Path



# |--------------------------------- Dev Toggles ----------------------------------|
# Used for dev, should be False in releases
SHOW_DEMO = True

# Card padding constants
CARD_OUT_X_PAD = 0.03   # padding L / R
CARD_OUT_Y_PAD = 0.06   # Padding top / bottom
CARD_GAP_X     = 0.03   # Between card padding L / R
CARD_GAP_Y     = 0.06   # Between card padding top / bottom

CARD_ROUNDING    = 0.06   # rounding relative to card size
CARD_INTERNAL_PAD = 5   # Padding between card content and edge

# Text size macros
SMALL_TEXT_SIZE  = 24.0
MEDIUM_TEXT_SIZE = 30.0
LARGE_TEXT_SIZE  = SMALL_TEXT_SIZE * 2
HEADER_TEXT_SIZE = SMALL_TEXT_SIZE * 3

SMALL_TEXT_PAD  = SMALL_TEXT_SIZE * 2
MEDIUM_TEXT_PAD = MEDIUM_TEXT_SIZE * 2
LARGE_TEXT_PAD  = LARGE_TEXT_SIZE * 2
# |--------------------------------- Dev Toggles ----------------------------------|


# |---------------------------------- Constants ----------------------------------|
CWD = Path(__file__).parent.parent
FONT_FOLDER = CWD / "font" / "League_Gothic" / "static"
# |---------------------------------- Constants ----------------------------------|


# |------------------------------ Internal Model ---------------------------------|
user_portfolio = None
rec_portfolios = []
# |------------------------------ Internal Model ---------------------------------|


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

    return outer_pad_x, outer_pad_y, gap_x, gap_y, card_w, card_h, rounding


def draw_card(card_w, card_h, rounding, label):
    with dpg.child_window(width=card_w, height=card_h, border=False):
        with dpg.drawlist(width=card_w, height=card_h):
          dpg.draw_rectangle(
              (0, 0),
              (card_w, card_h),
              color=(70, 70, 70, 255),
              fill=(35, 35, 35, 255),
              rounding=rounding,
              thickness=1,
          )
          dpg.draw_text(
                      (CARD_INTERNAL_PAD, CARD_INTERNAL_PAD),
                      text=label,
                      size=SMALL_TEXT_SIZE
                      )
          dpg.draw_text(
                      (CARD_INTERNAL_PAD, SMALL_TEXT_PAD + CARD_INTERNAL_PAD),
                      text="Widgets go here",
                      size=SMALL_TEXT_SIZE
                      )


def init_windows(w, h, fonts):
  """
  Inputs:
    - w    : total screen width
    - h    : total screen height
    - fonts: List of available fonts
  """
  screen_width  = w
  screen_height = h

  """
  The goal here was for the width of the portfolio window to be approximatiely
    2/3 of the screen. Unfortunately the current design opens to what tkinter thinks
    the screen size is, which for whatever reason seems to be a 9:16 AR or something
    on my laptop monitor. I'm not sure what's happening, but w + w/14 seems to fit
    well. Definitely want to do some more testing on my desktop to see what happens
    on different resolutions and monitors.
  """
  with dpg.window(
                tag="Portfolio Window",
                pos=(0, 0),
                width=(screen_width + (screen_width/14)),
                height=screen_height,
                no_move=True,
                no_title_bar=True
                ):
      # Any dpg window code in this block will apply to the above window.
      #   - We can later modify it by specifying parent="Portfolio Window"
      outer_pad_x, outer_pad_y, gap_x, gap_y, card_w, card_h, rounding = compute_layout(
          screen_width, screen_height
      )

      # Outer padding
      dpg.add_spacer(height=outer_pad_y)

      """
        U.S. Equities
        International Equities
        Fixed Income
      """
      with dpg.group(horizontal=True):
        dpg.add_spacer(width=((card_w * 2) / 5))
        us_equity_header  = dpg.add_text("U.S. Equities")

        dpg.add_spacer(width=((card_w * 5) / 8))
        int_equity_header = dpg.add_text("International Equities")

        dpg.add_spacer(width=((card_w * 5) / 9))
        fi_header         = dpg.add_text("Fixed Income")

        dpg.add_spacer(width=outer_pad_x)

      # Row 1 (3 cards), with left padding via spacer
      with dpg.group(horizontal=True):
          dpg.add_spacer(width=outer_pad_x)
          for i in range(3):
              with dpg.group():
                  draw_card(card_w, card_h, rounding, f"Cell {i+1}")
              if i < 2:
                  dpg.add_spacer(width=gap_x)

          dpg.add_spacer(width=outer_pad_x)

      dpg.add_spacer(height=gap_y)

      """
        Real Assets / REITs
        Cash / Cash equivalents
      """
      with dpg.group(horizontal=True):
          dpg.add_spacer(width=outer_pad_x + (card_w * 6) / 11)
          reit_header = dpg.add_text("Real Assets / REITs")

          dpg.add_spacer(width=gap_x + ((card_w * 14) / 19))
          cash_header = dpg.add_text("Cash / Cash Equivalents")
          dpg.add_spacer(width=(card_w/3) + outer_pad_x)

      # Row 2 (2 cards)
      with dpg.group(horizontal=True):
          dpg.add_spacer(width=outer_pad_x + card_w/3)
          for i in range(2):
              with dpg.group():
                  draw_card(card_w, card_h, rounding, f"Cell {i+4}")
              if i < 1:
                  dpg.add_spacer(width=gap_x)
              dpg.add_spacer(width=(card_w/3))

      dpg.add_spacer(width=outer_pad_x)

  # Bind all the headers in one pass to keep things consistent.
  for item in (us_equity_header, int_equity_header, fi_header, reit_header, cash_header):
    dpg.bind_item_font(item, font=fonts["large"])
# |----------------------- Screen Initialization Utilities ------------------------|


# |---------------------------- Screen Initialization -----------------------------|
# We load tkinter literally just to fetch the screen specs
def screen_init():
  root = tk.Tk()
  root.withdraw()
  screen_width = root.winfo_screenheight()
  screen_height = root.winfo_screenwidth()
  root.destroy()

  dpg.create_context()
  dpg.create_viewport(title='Portfolio Lab',
                      width=screen_width,
                      height=screen_height,
                      x_pos=0, y_pos=0,
                      )

  with dpg.font_registry():
      default_font = dpg.add_font(FONT_FOLDER / "LeagueGothic-Regular.ttf", SMALL_TEXT_SIZE)
      semicondensed_font = dpg.add_font(FONT_FOLDER / "LeagueGothic_SemiCondensed-Regular.ttf", MEDIUM_TEXT_SIZE)
      condensed_font = dpg.add_font(FONT_FOLDER / "LeagueGothic_Condensed-Regular.ttf", LARGE_TEXT_SIZE)

  dpg.bind_font(default_font)

  fonts = {
          "small": default_font,
          "medium": semicondensed_font,
          "large": condensed_font
          }

  # |---------------------------- DEFINE UI TREE HERE ----------------------------|
  init_windows(screen_width, screen_height, fonts)
  # |---------------------------- DEFINE UI TREE HERE ----------------------------|

  # Anything defined after these calls simply won't appear
  dpg.setup_dearpygui()
  dpg.show_viewport()
  dpg.maximize_viewport()
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
