# portfolio-lab
A portfolio allocation optimization app. Using a Markowitz portfolio optimizer trained by me, this app will determine an 'ideal' allocation of your financial assets for maximum gain at a given level of risk.

## Table of Contents

- [On Canaries & Coal Mines](#On-Canaries-&-Coal-Mines)
- [Installation](#installation)
- [Usage](#usage)
- [TODO](#TODO)
- [File Structure](#File-Structure)

## On Canaries & Coal Mines
* As I continue to work on this and attempt to improve it, I'm growing increasingly aware of the fact that this model simply may not outperform naive diversification. This section contains links to materials which are relevant to this realization.
  * https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901
    * Found that the gain from complex diversification optimization was significantly outweighed by mean estimation error.
    * Concludes that, for optimization gain to outweigh mean estimation error, a portfolio of 25 assets would need 3000 months of data, and a portfolio of 50 assets would need 6000 months of data.
      * Interpolating linearly, for my 5 asset classes this would imply we need 600 months, or 50 years of returns, we currently have 14 in the joined data.
      * Crucially, the language of the abstract implies they utilized monthly return values, which may effect the outcome.
  * https://www.researchaffiliates.com/publications/articles/821-is-diversification-dead
    * Argues that, while a strong case remains when examining long-horizon returns of diversified portfolios, undiversified portfolios have utterly dominated diversified portfolios in the 21st century.
    * Makes a strong argument for equally weighted ('pure diversification') outperforming a 60/40 weighted portfolio long-horizon
    * Contrarily to my experience, finds that a 60/40 portfolio performs better than an equally weighted portfolio, through the 2010s (short-horizon)
    * If this article is to be trusted, there's significant reason to believe that I *might* be able to show better-than-naive diversification gain. We would want data back to the 70s, necessitating a redefinition of our asset classes, since at least one originating asset began in 2011. This may be beyond scope.

## Installation

*empty for now*

## Usage

*empty for now*

## TODO

* Proper agnostic pathing
* requirements.txt for easy venv creation

## File Structure

```
.
├── documentation/
│   ├── Data Science Capstone Sources.docx
|   └── data.md
├── data/
|   ├── used/
|   └── raw/
├── src/
|   └── fetch_yahoo.py
└── README.md
```
