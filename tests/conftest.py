import pytest
import json

@pytest.fixture
def sample_equity_curve():
    return [
        50000, 50350, 50120, 50800, 51200, 50900, 51500, 51300,
        52000, 51600, 52400, 52100, 53000, 52700, 53500, 53200,
        54000, 53600, 54500, 55000
    ]

@pytest.fixture
def sample_pl_list():
    return [350.0, -120.0, 680.0, -200.0, -150.0, 1100.0, -300.0, -180.0, 420.0, -90.0]

@pytest.fixture
def sample_lean_output():
    return {
        "statistics": {
            "Total Orders": "10",
            "Net Profit": "22.043%",
            "Sharpe Ratio": "0.499",
            "Start Equity": "50000",
            "End Equity": "61021.56",
        },
        "profitLoss": {
            "2025-01-17T18:06:00Z": 353.58,
            "2025-01-21T14:33:00Z": 218.36,
            "2025-01-23T17:33:00Z": -589.12,
            "2025-02-03T14:33:00Z": 785.36,
            "2025-02-10T18:54:00Z": 317.44,
        },
        "charts": {
            "Strategy Equity": {
                "series": {
                    "Equity": {
                        "values": [
                            {"x": 1737000000, "y": 50000.0},
                            {"x": 1737100000, "y": 50353.58},
                            {"x": 1737200000, "y": 50571.94},
                            {"x": 1737300000, "y": 49982.82},
                            {"x": 1737400000, "y": 50768.18},
                            {"x": 1737500000, "y": 51085.62},
                        ]
                    }
                }
            }
        },
        "runtimeStatistics": {
            "Equity": "$61,021.56",
            "Net Profit": "$11,021.56",
            "Return": "22.04 %",
        }
    }
