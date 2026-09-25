"""
tests/test_population_fetcher.py — Unit test per il parser del Population Report
PriceCharting (poke_quant/data/population_fetcher.py).
"""

from unittest.mock import patch, Mock

from poke_quant.data.population_fetcher import parse_population_table, fetch_pricecharting_population

# Fixture reale: catturata via fetch live da
# https://www.pricecharting.com/pop/item/pokemon-base-set/charizard-4 (2026-09-25),
# troncata alle prime righe della tabella - la struttura (classi CSS, spaziatura) è
# quella effettiva della pagina, non inventata.
REAL_FIXTURE_HTML = """
    <div id="population-table-container">
        <h1>Charizard #4 Population Report</h1>
        <table id="population-table">
            <thead>
                <tr>
                    <th class="grade-col">Grade</th>
                    <th class="psa-col">PSA</th>
                    <th class="cgc-col">CGC</th>
                    <th class="total-col">Total</th>
                    <th class="price-col">Price</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="grade-col">1</td>
                    <td class="psa-col">4,096</td>
                    <td class="cgc-col">-</td>
                    <td class="total-col">4,096</td>
                    <td class="price-col">$326.19</td>
                </tr>
                <tr>
                    <td class="grade-col">9</td>
                    <td class="psa-col">8,455</td>
                    <td class="cgc-col">2</td>
                    <td class="total-col">8,457</td>
                    <td class="price-col">$2976.69</td>
                </tr>
                <tr>
                    <td class="grade-col">10</td>
                    <td class="psa-col">486</td>
                    <td class="cgc-col">4</td>
                    <td class="total-col">490</td>
                    <td class="price-col">$12275.00</td>
                </tr>
                <tr>
                    <td class="grade-col">Total</td>
                    <td class="psa-col">99,246</td>
                    <td class="cgc-col">9</td>
                    <td class="total-col">99,255</td>
                    <td class="price-col"></td>
                </tr>
            </tbody>
        </table>
    </div>
"""


def test_parses_real_fixture_rows():
    rows = parse_population_table(REAL_FIXTURE_HTML)
    assert [r["grade"] for r in rows] == ["1", "9", "10"]  # riga "Total" esclusa


def test_parses_missing_company_as_none_not_zero():
    rows = parse_population_table(REAL_FIXTURE_HTML)
    grade1 = next(r for r in rows if r["grade"] == "1")
    assert grade1["cgc_pop"] is None  # "-" != 0 copie: nessun dato, non zero
    assert grade1["psa_pop"] == 4096


def test_parses_thin_company_population_correctly():
    rows = parse_population_table(REAL_FIXTURE_HTML)
    grade10 = next(r for r in rows if r["grade"] == "10")
    assert grade10 == {"grade": "10", "psa_pop": 486, "cgc_pop": 4, "total_pop": 490, "price_usd": 12275.00}


def test_no_population_table_returns_empty_list():
    assert parse_population_table("<html><body>no data here</body></html>") == []


def test_fetch_returns_parsed_rows_on_200():
    mock_resp = Mock(status_code=200, text=REAL_FIXTURE_HTML)
    mock_resp.raise_for_status = Mock()
    with patch("poke_quant.data.population_fetcher.requests.get", return_value=mock_resp):
        rows = fetch_pricecharting_population("pokemon-base-set", "charizard-4")
    assert len(rows) == 3


def test_fetch_returns_empty_list_on_404_not_an_error():
    mock_resp = Mock(status_code=404, text="")
    with patch("poke_quant.data.population_fetcher.requests.get", return_value=mock_resp):
        rows = fetch_pricecharting_population("pokemon-base-set", "nonexistent-card")
    assert rows == []


def test_fetch_retries_after_429():
    resp_429 = Mock(status_code=429, headers={})
    resp_ok = Mock(status_code=200, text=REAL_FIXTURE_HTML)
    resp_ok.raise_for_status = Mock()
    with patch("poke_quant.data.population_fetcher.requests.get", side_effect=[resp_429, resp_ok]), \
         patch("poke_quant.data.population_fetcher.time.sleep"):
        rows = fetch_pricecharting_population("pokemon-base-set", "charizard-4")
    assert len(rows) == 3
