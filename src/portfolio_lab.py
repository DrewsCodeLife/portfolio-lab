"""
This is the main script file for portfolio lab.

|-----------| Ez access |-----------|
| STRING WIDGET |  |  INPUT WIDGET  |
us_equities_main     , us_equities_input     , us_equities_participation
int_equities_main    , int_equities_input    , int_equities_participation
fixed_income_main    , fixed_income_input    , fixed_income_participation
real_assets_main     , real_assets_input     , real_assets_participation
cash_equivalents_main, cash_equivalents_input, cash_equivalents_participation
"""


import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import tkinter as tk
import pandas as pd
import numpy as np
import cvxpy as cp

from sklearn.covariance import LedoitWolf
from scipy.linalg import solve
from pathlib import Path
from math import floor



np.random.seed(42)

# |--------------------------------- Dev Toggles ----------------------------------|
# Used for dev, should be False in releases
SHOW_DEMO = False

DEBUG_OUTPUT = True
VERBOSE_DEBUG_OUTPUT = True
# |--------------------------------- Dev Toggles ----------------------------------|


# |---------------------------------- Constants ----------------------------------|
# Card padding constants
CARD_OUT_X_PAD = 0.03   # padding L / R
CARD_OUT_Y_PAD = 0.06   # Padding top / bottom
CARD_GAP_X     = 0.03   # Between card padding L / R
CARD_GAP_Y     = 0.06   # Between card padding top / bottom

CARD_ROUNDING     = 0.06  # rounding relative to card size
CARD_INTERNAL_PAD = 5     # Padding between card content and edge

SIM_WINDOW_SIZE = float(1 / 3)

# Text size macros
SMALL_TEXT_SIZE  = 24.0
MEDIUM_TEXT_SIZE = 30.0
LARGE_TEXT_SIZE  = SMALL_TEXT_SIZE * 2
HEADER_TEXT_SIZE = SMALL_TEXT_SIZE * 3

SMALL_TEXT_PAD  = SMALL_TEXT_SIZE * 2
MEDIUM_TEXT_PAD = MEDIUM_TEXT_SIZE * 2
LARGE_TEXT_PAD  = LARGE_TEXT_SIZE * 2

TRADING_DAYS = 252

W_MIN = 0.00
# |---------------------------------- Constants ----------------------------------|


# |---------------------------------- App Class ----------------------------------|
class PortfolioLab:
    def __init__(self, width, height):
        # |------------------------------ Internal Model ---------------------------------|
        self.state = {
            "rec_portfolio": {  # default portfolio assumes flat distribution
                "U.S. Equities"             : 0.2,
                "International Equities"    : 0.2,
                "Fixed Income"              : 0.2,
                "Real Assets"               : 0.2,
                "Cash Equivalents"          : 0.2
                },
            "user_portfolio": {
                "U.S. Equities"             : 20.0,
                "International Equities"    : 20.0,
                "Fixed Income"              : 20.0,
                "Real Assets"               : 20.0,
                "Cash Equivalents"          : 20.0
            },
            # 1 Source of truth for column order (from daily_returns.csv)
            "asset_names": [
                "U.S. Equities",
                "International Equities",
                "Fixed Income",
                "Real Assets",
                "Cash Equivalents"
            ],
            "portfolio_value"   : 100.0,
            "has_run"           : False,
            "lookback"          : 252,
            "mu"                : None, # daily
            "sigma"             : None, # daily
            "req_return_daily"  : None, # Converted to daily from UI annualized request
        }
        # |------------------------------ Internal Model ---------------------------------|


        # |---------------------------------- Data Model ----------------------------------|
        self.CWD = Path(__file__).parent.parent
        self.FONT_FOLDER = self.CWD / "font" / "League_Gothic" / "static"

        self.data    = pd.read_csv("data/cleaned/cleaned.csv")
        self.returns = pd.read_csv("data/cleaned/daily_returns.csv")

        self.dates = self.data["Date"]

        self.data = self.data.loc[:, self.data.columns != "Date"]
        self.returns = self.returns.loc[:, self.returns.columns != "Date"]

        # First day has no 'daily return', so we drop it
        self.data = self.data.iloc[1:].reset_index(drop=True)
        self.returns = self.returns.iloc[1:].reset_index(drop=True)

        self.T = len(self.returns)
        self.N = self.returns.shape[1]

        self.screen_width = width
        self.screen_height = height
        # |---------------------------------- Data Model ----------------------------------|


        # |----------------------------- Window Initialization -----------------------------|
        # Drawing the window means we need fonts
        with dpg.font_registry():
            default_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic-Regular.ttf", SMALL_TEXT_SIZE)
            semicondensed_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic_SemiCondensed-Regular.ttf", MEDIUM_TEXT_SIZE)
            condensed_font = dpg.add_font(self.FONT_FOLDER / "LeagueGothic_Condensed-Regular.ttf", LARGE_TEXT_SIZE)

        dpg.bind_font(default_font)

        self.fonts = {
            "small": default_font,
            "medium": semicondensed_font,
            "large": condensed_font
            }

        with dpg.theme() as self.transparent_child_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                # Set background to transparent
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                # Remove border
                dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 0, category=dpg.mvThemeCat_Core)

        self._compute_layout()

        self._init_windows()

        # We've loaded the data and lookback defaults to 252, so this should be safe.
        #   We're going to want the initial values stored so we can clamp annualized returns
        self._estimate_mu_sigma()
        # |----------------------------- Window Initialization -----------------------------|


    # |----------------------------- Window Initialization -----------------------------|
    def _init_windows(self):
        """
        Inputs:
            - w    : total screen width
            - h    : total screen height
            - fonts: List of available fonts

        The goal here was for the width of the portfolio window to be approximatiely
            2/3 of the screen. Unfortunately the current design opens to what tkinter thinks
            the screen size is, which for whatever reason seems to be a 9:16 AR or something
            on my laptop monitor. I'm not sure what's happening, but w + w/14 seems to fit
            well. Definitely want to do some more testing on my desktop to see what happens
            on different resolutions and monitors.

        @TODO: Fix the screen resolution, it should be consistent and simple, not w + (w / arbitrary value)
        """
        with dpg.window(
            tag="Portfolio Window",
            pos=(0, 0),
            width=(self.screen_width + (self.screen_width / 14)),
            height=self.screen_height,
            no_move=True,
            no_title_bar=True
            ):
            # Any dpg window code in this block will apply to the above window.
            #   - We can later modify it by specifying parent="Portfolio Window"

            # Outer padding
            dpg.add_spacer(height=self.outer_pad_y)

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=((self.card_w * 2) / 5))
                us_equity_header  = dpg.add_text("U.S. Equities")

                dpg.add_spacer(width=((self.card_w * 5) / 8))
                int_equity_header = dpg.add_text("International Equities")

                dpg.add_spacer(width=((self.card_w * 5) / 9))
                fi_header         = dpg.add_text("Fixed Income")

                dpg.add_spacer(width=self.outer_pad_x)

            # Row 1 (3 cards), with left padding via spacer
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=self.outer_pad_x)

                pf_val = self.state['portfolio_value']

                with dpg.group():
                    in_val = self.state['user_portfolio']['U.S. Equities']
                    rec_val = pf_val * self.state['rec_portfolio']['U.S. Equities']
                    change, high = self._overunder(in_val, rec_val)

                    self._draw_card(
                        tag="us_equities",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        change=change,
                        high=high,
                        callback=self._us_class_update
                        )
                dpg.add_spacer(width=self.gap_x)

                in_val = self.state['user_portfolio']['International Equities']
                rec_val = pf_val * self.state['rec_portfolio']['International Equities']
                change, high = self._overunder(in_val, rec_val)

                with dpg.group():
                    self._draw_card(
                        tag="int_equities",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        change=change,
                        high=high,
                        callback=self._inter_class_update
                        )
                dpg.add_spacer(width=self.gap_x)

                in_val = self.state['user_portfolio']['Fixed Income']
                rec_val = pf_val * self.state['rec_portfolio']['Fixed Income']
                change, high = self._overunder(in_val, rec_val)

                with dpg.group():
                    self._draw_card(
                        tag="fixed_income",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        change=change,
                        high=high,
                        callback=self._fixed_class_update
                        )

                dpg.add_spacer(width=self.outer_pad_x)

            dpg.add_spacer(height=self.gap_y)

            with dpg.group(horizontal=True):
                dpg.add_spacer(width=self.outer_pad_x + (self.card_w * 6) / 11)
                reit_header = dpg.add_text("Real Assets / REITs")

                dpg.add_spacer(width=self.gap_x + ((self.card_w * 14) / 19))
                cash_header = dpg.add_text("Cash / Cash Equivalents")
                dpg.add_spacer(width=(self.card_w/3) + self.outer_pad_x)

            # Row 2 (2 cards)
            with dpg.group(horizontal=True):
                dpg.add_spacer(width=self.outer_pad_x + (self.card_w / 3))

                in_val = self.state['user_portfolio']['Real Assets']
                rec_val = pf_val * self.state['rec_portfolio']['Real Assets']
                change, high = self._overunder(in_val, rec_val)

                with dpg.group():
                    self._draw_card(
                        tag="real_assets",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        change=change,
                        high=high,
                        callback=self._reits_class_update
                        )
                dpg.add_spacer(width=(self.gap_x + (self.card_w / 3)))

                in_val = self.state['user_portfolio']['Cash Equivalents']
                rec_val = pf_val * self.state['rec_portfolio']['Cash Equivalents']
                change, high = self._overunder(in_val, rec_val)

                with dpg.group():
                    self._draw_card(
                        tag="cash_equivalents",
                        label=f"IN: ${in_val} / REC: ${rec_val:.2f}",
                        change=change,
                        high=high,
                        callback=self._cash_class_update
                        )

            dpg.add_spacer(width=self.outer_pad_x)

        # Bind all the headers in one pass to keep things consistent.
        for item in (us_equity_header, int_equity_header, fi_header, reit_header, cash_header):
            dpg.bind_item_font(item, font=self.fonts["large"])

        with dpg.window(
            pos=(self.screen_width + (self.screen_width/14) + (self.outer_pad_x / 2), 0),
            width=(self.screen_width * (3 / 4)),
            height=self.screen_height,
            no_move=True,
            no_title_bar=True
            ):
            with dpg.group():
                with dpg.drawlist(
                    width=(self.screen_width * (3 / 4)),
                    height=self.screen_height * SIM_WINDOW_SIZE
                    ):
                    # Here we can define a sim plot window
                    dpg.draw_rectangle(
                        (0, 0),
                        ((self.screen_width * (3 / 4)), self.screen_height * SIM_WINDOW_SIZE),
                        color=(70, 70, 70, 255),
                        fill=(35, 35, 35, 255),
                        rounding=self.rounding,
                        thickness=1,
                        )

                # Here we can define the manual data entry
                dpg.add_slider_int(
                    label="Lookback period (days)",
                    tag="lookback",
                    min_value=252,
                    max_value=(self.T - 252),
                    callback=self._lookback_update,
                    user_data=self.T,
                    default_value=252
                    )
                dpg.add_slider_int(
                    label=f"Simulation Period [1, {floor(self.T / 252)}] (years)",
                    tag="sim_period",
                    min_value=1,
                    max_value=floor(self.T / 252),
                    user_data=self.T,
                    default_value=1,
                    callback=self._sim_period_update,
                    )
                dpg.add_slider_float(
                    label="Annualized Growth Percentage",
                    tag="desired_ret",
                    min_value=0.01,
                    max_value=0.2,
                    default_value=1.0,
                    callback=self._update_desired_ret,
                    )
                dpg.add_text(default_value=self.state['portfolio_value'], tag='pv_value')
                dpg.add_button(label="Optimize Portfolio", callback=self._construct_portfolio)


    def _compute_layout(self):
        self.outer_pad_x = int(self.screen_width * CARD_OUT_X_PAD)
        self.outer_pad_y = int(self.screen_height * CARD_OUT_Y_PAD)
        self.gap_x       = int(self.screen_width * CARD_GAP_X)
        self.gap_y       = int(self.screen_height * CARD_GAP_Y)

        # Card width from 3 columns
        self.card_w = (self.screen_width - 2 * self.outer_pad_x - 2 * self.gap_x) / 3.0

        # Card height from 2 rows
        self.card_h = (self.screen_height - 2 * self.outer_pad_y - 1 * self.gap_y) / 2.0

        # Enforce card slightly wider than tall
        self.card_h = min(self.card_h, self.card_w * 0.85)

        self.card_w = int(self.card_w)
        self.card_h = int(self.card_h)

        self.rounding = int(min(self.card_w, self.card_h) * CARD_ROUNDING)


    def _draw_card(self, tag, label, change, high, callback):
        with dpg.child_window(
            tag=tag + '_window_1',
            width=self.card_w,
            height=self.card_h,
            no_scrollbar=True,
            no_scroll_with_mouse=True,
            ):
            with dpg.drawlist(
                width=self.card_w,
                height=self.card_h,
                ):
                dpg.draw_rectangle(
                    (0, 0),
                    (self.card_w, self.card_h),
                    color=(70, 70, 70, 255),
                    fill=(35, 35, 35, 255),
                    rounding=self.rounding,
                    thickness=1,
                )


            with dpg.child_window(
                pos=(0, 0),  # Draw text over rectangle
                tag=tag + '_window_2',
                width=self.card_w,
                height=self.card_h,
                border=False,
                ):

                with dpg.group():
                    dpg.add_text(
                        default_value=label,
                        pos=(CARD_INTERNAL_PAD, CARD_INTERNAL_PAD),
                        tag=tag + "_main",
                        )

                    bl_label = ''
                    if high is None:
                        # Balanced (+/- 5%)
                        bl_label = f"ASSET CLASS BALANCED!"
                        pass
                    elif high:
                        # More than 5% over contributed
                        bl_label = f"OVERPARTICIPATING: {change:.3f}%"
                        pass
                    else:
                        # Less then 95% under contributed
                        bl_label = f"UNDERPARTICIPATING: {change:.3f}%"

                    dpg.add_text(
                        default_value=bl_label,
                        pos=(CARD_INTERNAL_PAD, SMALL_TEXT_PAD + CARD_INTERNAL_PAD),
                        tag=tag + '_participation'
                    )

                    # step & step_fast = 0 disables +/- buttons
                    dpg.add_input_double(
                        pos=(CARD_INTERNAL_PAD, (SMALL_TEXT_PAD * 2) + CARD_INTERNAL_PAD),
                        tag=tag + "_input",
                        label="Current Portfolio Value",
                        width=self.card_w / 3,
                        step=0,
                        step_fast=0,
                        default_value=0.0,
                        callback=callback,
                        )

                dpg.bind_item_font(tag + "_main", font=self.fonts["small"])
                dpg.bind_item_font(tag + '_participation', font=self.fonts["small"])

            dpg.bind_item_theme(tag + '_window_1', self.transparent_child_theme)
            dpg.bind_item_theme(tag + '_window_2', self.transparent_child_theme)
    # |----------------------------- Window Initialization -----------------------------|


    # |----------------------------- Callback Functions --------------------------------|
    def _lookback_update(self, sender, app_data, user_data):
        self.state['lookback'] = lookback = app_data
        sim_max = floor((user_data - lookback) / 252)

        if (dpg.get_value("sim_period") > sim_max):
            dpg.set_value("sim_period", sim_max)

        dpg.configure_item("sim_period", max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")

        # Need to recalculate, lookback changed
        self._estimate_mu_sigma()

        # Since mu has changed, so have our clamps!
        self.clamp_returns()


    def _sim_period_update(self, sender, app_data, user_data):
        cur_lookback = dpg.get_value("lookback")
        sim_max = floor((user_data - cur_lookback) / 252)
        if (app_data > sim_max):
            with dpg.popup(dpg.get_item_parent(sender)):
                dpg.add_text("ERROR: Requested sim period exceeds available days - lookback period")
                dpg.set_value(sender, 1)

        dpg.configure_item(sender, max_value=sim_max, label=f"Simulation Period [1, {sim_max}] (years)")


    def _us_class_update(self, sender, app_data):
        # To keep the portfolio value synchronized, reduce by old value, increase by new value
        old_pv = self.state['portfolio_value']
        new_pv = old_pv - self.state['user_portfolio']['U.S. Equities'] + app_data
        self.state['portfolio_value'] = new_pv

        self.state['user_portfolio']['U.S. Equities'] = app_data

        dpg.configure_item('pv_value', default_value=new_pv)

        rec_val = new_pv * self.state['rec_portfolio']['U.S. Equities']

        change, high = self._overunder(app_data, rec_val)

        if high is None:
            dpg.configure_item('us_equities_participation', default_value=f"ASSET CLASS BALANCED!")
        elif high:
            dpg.configure_item('us_equities_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
        else:
            dpg.configure_item('us_equities_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

        dpg.configure_item("us_equities_main", default_value=f"IN: ${app_data} / REC: ${rec_val:.2f}")

        self._update_all_card_states()


    def _inter_class_update(self, sender, app_data):
        self.state['user_portfolio']['International Equities'] = app_data
        self._update_all_card_states()


    def _fixed_class_update(self, sender, app_data):
        self.state['user_portfolio']['Fixed Income'] = app_data
        self._update_all_card_states()


    def _reits_class_update(self, sender, app_data):
        self.state['user_portfolio']['Real Assets'] = app_data
        self._update_all_card_states()


    def _cash_class_update(self, sender, app_data):
        self.state['user_portfolio']['Cash Equivalents'] = app_data
        self._update_all_card_states()

    def _update_desired_ret(self, sender, app_data):
        # Convert expected linear annual return to daily linear return, convert to decimal
        self.state['req_return_daily'] = app_data / (252 * 100)

    def _construct_portfolio(self, sender, app_data):
        if DEBUG_OUTPUT:
            print("\n\n\n\n\n\nNEW OPTIMIZATION BEGINNING")

        pf = self._build_long_only_portfolio(w_max=None)

        asset_names = self.state['asset_names']

        if pf is None:
            # Optimization failed
            # Collapse further
            return

        if pf['status'] == cp.OPTIMAL:
            print("Portfolio Solved:", pf['exp_return_ann'], pf['vol_ann'])
            print(f"Solved weights:\n"
                    + "\n".join(f"{name}: {pf['weights'][i]}" for i, name in enumerate(asset_names))
            )
        else:
            print("Target return unsolvable. Or potentially invalid R_target. Future pull up popup. For now crash")
            exit(-1)
    # |----------------------------- Callback Functions --------------------------------|


    # |------------------------------ Helper Functions ---------------------------------|
    # |------------------------------- INTERNALIZED ----------------------------------|
    def _overunder(self, in_val, rec_val):
        if in_val is None:
            in_val = rec_val

        change = -1
        high = None
        if in_val > rec_val * 1.05:
            # change - int(change) = inverse truncation.
            # * 100 = percentage increase from rec_val to in_val
            change = ((in_val - rec_val) / rec_val) * 100
            high = True
        elif in_val < rec_val * 0.95:
            change = ((rec_val - in_val) / rec_val) * 100
            high = False
        return change, high


    def _update_all_card_states(self):
        pf_ref = self.state['user_portfolio']
        rec_ref = self.state['rec_portfolio']
        pf_val = self.state['portfolio_value'] = sum(self.state['user_portfolio'].values())

        dpg.configure_item('pv_value', default_value=pf_val)

        # |------------------------------ U.S. Equities ------------------------------|
        in_val = pf_ref['U.S. Equities']
        rec_val = pf_val * rec_ref['U.S. Equities']
        dpg.configure_item('us_equities_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ U.S. Equities ------------------------------|

        # |------------------------- International Equities -------------------------|
        in_val = pf_ref['International Equities']
        rec_val = pf_val * rec_ref['International Equities']

        change, high = self._overunder(in_val, rec_val)

        if high is None:
            dpg.configure_item('int_equities_participation', default_value=f"ASSET CLASS BALANCED!")
        elif high:
            dpg.configure_item('int_equities_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
        else:
            dpg.configure_item('int_equities_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

        dpg.configure_item('int_equities_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------- International Equities -------------------------|

        # |------------------------------ Fixed Income ------------------------------|
        in_val = pf_ref['Fixed Income']
        rec_val = pf_val * rec_ref['Fixed Income']

        change, high = self._overunder(in_val, rec_val)

        if high is None:
            dpg.configure_item('fixed_income_participation', default_value=f"ASSET CLASS BALANCED!")
        elif high:
            dpg.configure_item('fixed_income_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
        else:
            dpg.configure_item('fixed_income_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

        dpg.configure_item('fixed_income_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ Fixed Income ------------------------------|

        # |------------------------------ Real Assets ------------------------------|
        in_val = pf_ref['Real Assets']
        rec_val = pf_val * rec_ref['Real Assets']

        change, high = self._overunder(in_val, rec_val)

        if high is None:
            dpg.configure_item('real_assets_participation', default_value=f"ASSET CLASS BALANCED!")
        elif high:
            dpg.configure_item('real_assets_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
        else:
            dpg.configure_item('real_assets_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

        dpg.configure_item('real_assets_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |------------------------------ Real Assets ------------------------------|

        # |--------------------------- Cash Equivalents ---------------------------|
        in_val = pf_ref['Cash Equivalents']
        rec_val = pf_val * rec_ref['Cash Equivalents']

        change, high = self._overunder(in_val, rec_val)

        if high is None:
            dpg.configure_item('cash_equivalents_participation', default_value=f"ASSET CLASS BALANCED!")
        elif high:
            dpg.configure_item('cash_equivalents_participation', default_value=f"OVERPARTICIPATING: {change:.3f}%")
        else:
            dpg.configure_item('cash_equivalents_participation', default_value=f"UNDERPARTICIPATING: {change:.3f}%")

        dpg.configure_item('cash_equivalents_main', default_value=f"IN: ${in_val} / REC: ${rec_val:.2f}")
        # |--------------------------- Cash Equivalents ---------------------------|


    def _estimate_mu_sigma(self):
        """
        Estimate mean values and covariance matrix of daily returns
        """
        lookback = self.state['lookback']

        if lookback is not None:
          local_data = self.returns.iloc[-lookback:]
        else:
          # @TODO Error handle
          print("Error: Lookback was None")
          pass

        mu = local_data.mean().values.reshape(-1, 1) # (5x1 vector, reshaped for matrix mult)

        # The original script allowed shrinking or not. My understanding is we pretty much always
        #   want to, so I've collapsed the conditional to a single path.
        lw = LedoitWolf().fit(local_data.values)
        sigma = lw.covariance_

        # This is a covariance matrix, it'd better already be symmetric.
        # Nonetheless, we explicitly make it symmetric for safety.
        # - If it weren't symmetric, there could be multiple local optima,
        #     And I'm pretty sure the solver would do this anyway.
        sigma = (sigma + sigma.T) / 2
        self.state['mu'] = np.asarray(mu).reshape(-1) # We most often use it in this form
        self.state['sigma'] = sigma


    def _build_long_only_portfolio(self, w_max=None):
        """
        Given a target return (daily) that the user wants, provide an 'optimized' portfolio

        Important detail:
        - With long-only constraints, not every target return is feasible.
        - The feasible expected-return range is roughly [min(mu), max(mu)] under long-only,
        but in practice the covariance + sum-to-1 can still make some exact equality targets
        numerically finicky.
        - So we generate targets inside a "safe" range and skip infeasible solves.

        Solves:
            minimize      : w^T Sigma w
            Constrained by: sum(w) = 1
                            w >= 0
                            mu^T w = R_target
                            (optional) w <= w_max

        Parameters
        ----------
        w_max : float or (N,) array, optional
            Weight caps.
        solver : str
            Solver to use. OSQP is just as good here as it was for min var

        Returns
        -------
        portfolio : dict
            Contains weights, target return, realized return/vol, status.
        """
        mu = self.state['mu']
        Sigma = self.state['sigma']

        # Commented out for now. Probably warn the user (with explanation) and err out
        #   if it exceeds like 1e5 or something
        if DEBUG_OUTPUT:
            cond_number = np.linalg.cond(Sigma)
            eigs = np.linalg.eigvals(Sigma)
            print(f"Condition number: {cond_number}")
            print(f"Min Eigenvalue: {min(eigs)}\nMax Eigenvalue: {max(eigs)}")

        N = Sigma.shape[0]

        w = cp.Variable(N)

        objective = cp.Minimize(cp.quad_form(w, Sigma))

        R_target = self.state['req_return_daily']

        if R_target is None:
            # The user hasn't selected a value yet.
            #   - Probably notify via popup and abort optimization
            print("ERROR: R_target uninitialized")
            return None

        # Trying to minimize the work for OSQP to get it to work hopefully
        mu = np.asarray(mu, dtype=float).reshape(N,)
        R_target = float(R_target)

        constraints = [
            cp.sum(w) == 1,          # fully invested
            w >= W_MIN,              # long-only, minimum {W_min*100}% participation
            (-mu) @ w <= -R_target   # target return or larger
        ]

        if DEBUG_OUTPUT:
            print("We're asking cvxpy to solve the following problem:")
            print(f"mu: {mu}\nsigma: {Sigma}\nTarget Daily Return: {R_target}")

        if w_max is not None:
            constraints.append(w <= w_max)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver="SCS", verbose=False)

        # Fail early if optiimization was unsuccessful
        if w.value is None:
            if DEBUG_OUTPUT:
                print(f"Optimization was {problem.status}")
            if VERBOSE_DEBUG_OUTPUT:
                print(f"Solver statistics: {problem.solver_stats}")
            return None

        w = np.asarray(w.value).ravel()
        status = problem.status

        # 'asset_names' ensures columns remain aligned
        # I could probably use the dict itself, it's in the right order,
        #   but this is a better design pattern (1 source of truth)
        self.state['rec_portfolio'] = dict(zip(self.state['asset_names'], w))
        stats = self._compute_portfolio_stats(annualize=True)

        # Call update functions, new rec values
        self._update_all_card_states()

        self.state['markowitz_status'] = status

        return {
            "target_R": float(R_target),
            "status": status,
            "weights": w,
            **stats
        }


    def _compute_portfolio_stats(self, annualize=True):
        """
        Utility helper: compute expected return and volatility for weights w.

        mu: expected daily returns
        Sigma: daily covariance
        w: weights

        Returns dict with daily and (optionally) annualized metrics.
        """
        mu = self.state['mu']
        Sigma = self.state['sigma']
        w = np.asarray(list(self.state['rec_portfolio'].values()), dtype=float).reshape(-1)

        exp_ret_daily = float(mu @ w)
        var_daily = float(w.T @ Sigma @ w)
        vol_daily = float(np.sqrt(max(var_daily, 0.0)))

        out = {
            "exp_return_daily": exp_ret_daily,
            "vol_daily": vol_daily,
        }

        if annualize:
            out["exp_return_ann"] = exp_ret_daily * TRADING_DAYS
            out["vol_ann"] = vol_daily * np.sqrt(TRADING_DAYS)

        return out
    # |------------------------------- INTERNALIZED ----------------------------------|


    # |------------------------------- EXTERNALIZED ----------------------------------|
    def clamp_returns(self):
        """
        Externalized because we want to call it from main, after the widget exists
            But before it's visible
        """
        mu = self.state['mu']

        mu_min = float(mu.min())
        mu_max = float(mu.max())

        # Clamping the range of requestable returns to the (approximate) minimum and maximum return
        #   from the data. +/- a slight offset to ensure the request is solvable.
        eps = 1e-12

        # Harder to hit max than min, we clamp max by 50% more than min
        R_max = (mu_max - eps * (3/2)) * 252 * 100
        R_min = (mu_min + eps) * 252 * 100

        # Only runs on initialization of the app, used if user tries to optimize a portfolio
        #   without having supplied a desired return first
        self.state['req_return_daily'] = R_min / (252 * 100)

        # * 252 to annualize, expected linear return (compounding ignored)
        # * 100 so we see percent growth.
        dpg.configure_item(
            "desired_ret",
            min_value=R_min,
            max_value=R_max,
            default_value=R_min
            )
    # |------------------------------- EXTERNALIZED ----------------------------------|
    # |----------------------------- Helper Functions --------------------------------|
# |---------------------------------- App Class ----------------------------------|


# |-------------------------------- Main Function ---------------------------------|
def main():
    # |---------------------------- Screen Initialization -----------------------------|
    # We load tkinter literally just to fetch the screen specs
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenheight()
    height = root.winfo_screenwidth()
    root.destroy()

    dpg.create_context()
    dpg.create_viewport(
        title='Portfolio Lab',
        width=width,
        height=height,
        x_pos=0, y_pos=0,
        )
    # |---------------------------- Screen Initialization -----------------------------|

    pl = PortfolioLab(width, height)

    # Anything defined after these calls simply won't appear
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.maximize_viewport()

    if SHOW_DEMO:
        # Demo only shows up *inside* of the viewport we defined for the main window
        demo.show_demo()

    # This is our last chance to do anything, actions after this point won't populate the UI before it shows
    pl.clamp_returns()

    # This replaces dpg.start_dearpygui()
    while dpg.is_dearpygui_running():
        # This will run every frame
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
