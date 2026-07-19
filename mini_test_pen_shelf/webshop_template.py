"""Frozen, non-oracle WebShop playbook used by the standalone mini test.

The playbook only asks the model to reason over the task, current raw page and
the visible action/observation history.  It contains no goal index, target ASIN
or programmatically derived progress fields.
"""


MAX_SCORE_WEBSHOP_PLAYBOOK = r"""
## WebShop Max-Score State Machine (hard limit: 15 steps)

Keep four ledgers from the task and action history: PRODUCT TYPE, REQUIRED
ATTRIBUTES, REQUIRED OPTIONS, MAX PRICE; plus VISITED products, USED queries,
and OPTIONS ALREADY CLICKED. Follow exactly one matching page rule:

1. SEARCH PAGE: first search once for the exact product-type phrase + at most
   two rare attributes. Omit price and ordinary size/color. If refinement is
   truly needed, add the most unusual exact option/attribute; never reuse a
   query. On a results page, `Back to Search` means abandoning those results:
   never click it immediately after a search. Use it only to submit one
   genuinely different query after current results have been inspected.

2. RESULTS PAGE: before Next or Back, open a NEW in-budget candidate on the
   current page. Ranking is useful but synthetic titles/categories can be
   misleading: do not reject a low-price candidate solely because its title
   sounds unlike the request, especially when the task has exact options. Read
   the visible history before choosing an ASIN: an ASIN followed by `< prev>`
   was already rejected, so choose a different current ASIN. Never alternate
   Next and `< prev>` without opening a new product, and never return to Search
   without inspecting a new candidate.

3. PRODUCT PAGE WITH REQUESTED OPTION TEXTS: any explicit requested option on
   the page is stronger evidence than a surprising synthetic title/category.
   Do not leave this page merely because the title sounds wrong. For each
   required option shown on the page, click its exact text ONCE. Consult visible
   action history: never click an option already clicked on this product.
   Duplicate-looking option labels still require only one click. When all
   requested option texts visible on the page have been clicked and price fits,
   click Buy Now immediately; do not open Description or `< prev>`. After an
   exact option click, do not require the title to repeat that option value.

4. PRODUCT PAGE WITHOUT OPTIONS: if price fits and the title states the product
   type or two requested rare attributes, Buy Now. Do not require every use
   case/filler word to appear in the title. Only if one indispensable attribute
   is genuinely hidden, inspect Description or Features once.

5. DESCRIPTION/FEATURES PAGE: the ONLY navigation back to the product is
   click[< prev]. Never click Back to Search here. After returning, either Buy
   Now or reject the product once.

6. DEADLINE/LOOP RULE: target purchase by step 6. By step 8, buy the best
   in-budget candidate already opened (correct type/options and most
   attributes), because partial reward beats a zero-score timeout. A new search
   is justified only when it adds a new rare term; otherwise inspect or buy a
   current candidate. Never repeat an unchanged action, query, option click, or
   Search→same product→Back loop.

Every click must copy a currently admissible string exactly. Output exactly one
<think>...</think> then one <action>search[...]</action> or
<action>click[...]</action>, with no other text.
""".strip()
