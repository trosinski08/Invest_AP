🚀 Autonomous AI Investment Agent (MVP)
Opis projektu:
Autonomiczny agent finansowy, który łączy analizę sentymentu rynkowego (newsy) z analizą techniczną (dane giełdowe), aby podejmować samodzielne decyzje inwestycyjne. Agent działa w pętli, wykonując transakcje przez API brokera z rygorystycznym systemem zabezpieczeń (Circuit Breaker).

🛠 Wymagania Techniczne (System Requirements)
Aby agent był skuteczny i bezpieczny, musi spełniać cztery fundamenty techniczne:

1. Percepcja (Data Ingestion)
Market Data API: Dostęp do świec cenowych (OHLCV). Rekomendowane: yfinance (akcje) lub ccxt (krypto).

News/Search API: Narzędzie do skanowania sieci. Rekomendowane: DuckDuckGo Search Tool lub Serper.dev.

Context Window: LLM musi mieć min. 128k tokenów kontekstu (np. GPT-4o-mini lub Gemini 1.5 Flash), aby przetworzyć wiele nagłówków newsów naraz.

2. Rozumowanie (Reasoning Engine)
Model: GPT-4o lub Gemini 1.5 Pro (modele o wysokiej zdolności rozumowania logicznego).

Framework: LangChain lub LangGraph (do obsługi cykliczności i pamięci krótkotrwałej agenta).

Prompt Engineering: System prompt wymuszający formatowanie wyjściowe w JSON (do poprawnego parsowania decyzji przez kod).

3. Egzekucja i Bezpieczeństwo (The Guardrails)
Broker API: Klucze API z uprawnieniami Trade (BEZ uprawnień Withdraw).

Circuit Breaker: Niezależny moduł w Pythonie (poza LLM), który sprawdza:

Max Daily Loss: Stop po stracie X%.

Max Position Size: Zakaz otwierania pozycji większej niż Y% portfela.

Cooldown: Minimalny odstęp czasu między transakcjami (np. 1h).

4. Monitoring (Observability)
Logging: Zapisywanie każdego promptu i każdej decyzji do pliku .log.

Dashboard: Prosty interfejs w Streamlit do podglądu aktualnego stanu portfela i logiki agenta w czasie rzeczywistym.

📂 Struktura Projektu

```
├── app.py              # Interfejs Streamlit (Frontend / Dashboard)
├── main.py             # Główna pętla agenta (Autonomy Loop)
├── config.py           # Parametry ryzyka, modelu i rynku
├── requirements.txt    # Zależności Pythona
├── .env.example        # Szablon zmiennych środowiskowych
├── .env                # Twoje klucze API (NIE COMMITUJ!)
├── engine/
│   ├── __init__.py
│   ├── agent.py        # Logika LLM i promptowanie (OpenAI GPT)
│   ├── tools.py        # Funkcje: dane OHLCV (ccxt), newsy (DDG), wskaźniki (ta)
│   └── guardrails.py   # Circuit Breaker — sztywne limity bezpieczeństwa
└── data/
    ├── trades.log      # Historia decyzji i transakcji (JSONL)
    ├── paper_state.json# Stan portfela paper-trading
    └── agent.log       # Logi pracy agenta
```
📝 Roadmapa (Weekend Sprint)
Sobota: Budowa "Mózgu" i "Oczu"
[ ] Konfiguracja środowiska (Python 3.10+, venv).

[ ] Implementacja narzędzi do pobierania cen i newsów.

[ ] Stworzenie systemu promptów (Analiza techniczna + Sentyment).

[ ] Testy "na sucho" (LLM generuje decyzję, ale nie wysyła zlecenia).

Niedziela: Autonomia i Ryzyko
[ ] Połączenie z API brokera (tryb Paper Trading lub minimalny kapitał).

[ ] Zakodowanie modułu guardrails.py (sztywne limity straty).

[ ] Uruchomienie pętli autonomicznej (while True).

[ ] Budowa dashboardu w Streamlit do monitoringu "na żywo".

⚠️ DISCLAIMER: Ten projekt służy wyłącznie celom edukacyjnym. Handel na rynkach finansowych wiąże się z wysokim ryzykiem utraty kapitału. Autor nie bierze odpowiedzialności za decyzje podjęte przez AI.

---

## 🏁 Szybki start

```bash
# 1. Zainstaluj zależności
pip install -r requirements.txt

# 2. Skopiuj szablon kluczy i uzupełnij go
cp .env.example .env
# (edytuj .env — uzupełnij OPENAI_API_KEY, opcjonalnie Binance)

# 3. Uruchom agenta (paper trading)
python main.py

# 4. W osobnym terminalu — uruchom dashboard
streamlit run app.py
```

### Limity bezpieczeństwa (domyślne)

| Parametr | Wartość | Opis |
|---|---|---|
| MAX_ORDER_VALUE_USD | 20 USD | Max wartość jednego zlecenia |
| MAX_DAILY_LOSS_USD | 40 USD | Bot wyłącza się po tej stracie |
| MAX_TRADES_PER_24H | 5 | Limit transakcji na dobę |
| COOLDOWN_MINUTES | 60 min | Min. przerwa między transakcjami |
| STOP_LOSS_PCT | 3% | Automatyczny stop-loss |
| TAKE_PROFIT_PCT | 5% | Automatyczny take-profit |
| SENTIMENT_THRESHOLD | 0.7 | Minimalna pewność AI do handlu |