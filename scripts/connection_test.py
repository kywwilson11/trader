"""Alpaca API connection test — verifies credentials and reports account status.

Run standalone to check:
  - API key / secret are valid
  - Account equity, buying power, and trading_blocked status
  - Whether intraday margin applies (>= $2,000 equity; PDT retired June 2026)
"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from dotenv import load_dotenv

load_dotenv()


def get_trading_status():
    # Connect via the shared constructor (legacy SDK or alpaca-py adapter)
    from trading_utils import get_api
    api = get_api()

    try:
        # 1. Get Account Info
        account = api.get_account()

        # 2. Check Financials.
        # NOTE: The PDT rule was RETIRED June 4, 2026 (FINRA intraday-margin
        # framework). Alpaca deletes pattern_day_trader / daytrade_count from
        # the account API on July 6, 2026 — do not read them. A margin
        # account with >= $2,000 equity gets intraday buying power; cash
        # accounts remain bound by T+1 settled funds.
        equity = float(account.equity)
        buying_power = float(getattr(account, 'buying_power', 0) or 0)

        print(f"\n--- ACCOUNT STATUS ---")
        print(f"Equity:        ${equity:,.2f}")
        print(f"Buying Power:  ${buying_power:,.2f}")
        print(f"Status:        {account.status}")
        if getattr(account, 'trading_blocked', False):
            print("WARNING: trading_blocked is set on this account!")
            return "HALT"

        # 3. The Logic Gate (post-PDT world)
        if equity >= 2000:
            print("\n[DECISION]: OK TO TRADE")
            print("Margin account >= $2,000: intraday margin framework applies "
                  "(no day-trade count limits).")
            return "SAFE_TO_TRADE"
        else:
            print("\n[DECISION]: UNDERFUNDED")
            print("Equity under $2,000 — intraday margin unavailable; "
                  "a cash account is limited to settled funds (T+1).")
            return "CONSERVATIVE"

    except Exception as e:
        print(f"Error connecting to Alpaca: {e}")
        return "ERROR"

if __name__ == "__main__":
    # Run the check
    status = get_trading_status()
