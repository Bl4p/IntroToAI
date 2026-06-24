# TG-PFPKE Handoff Document

## Project Overview
A **Telemetry-Gated Privacy-Friendly Public Key Encryption** system with biometric authentication, AI risk evaluation, and a graphical web interface.

## Architecture

### Core Modules (unchanged)
| Module | Purpose |
|--------|---------|
| `biometric_scanner.py` | Simulates fingerprint scans, aligns images, extracts fuzzy picture (μ, η, ν vectors) |
| `pfpke_crypto.py` | Key generation from fuzzy data, Fernet symmetric encryption, Ed25519 signatures |
| `telemetry_db.py` | SQLite DB for enrollments/login_attempts, captures IP/geo/device telemetry |

### Risk Engines (3 options, selectable at runtime)
| Engine | File | Type | Parameters |
|--------|------|------|------------|
| Transformer | `tg_risk_model.py` | DL (Transformer Encoder) | ~thousands |
| GRU | `tg_risk_gru.py` | DL (Gated Recurrent Unit) | 1,201 |
| Isolation Forest | `tg_risk_iforest.py` | Classical ML (sklearn) | N/A |

### Training Scripts
| Script | Trains |
|--------|--------|
| `train_risk_engine.py` | Transformer risk model |
| `train_risk_gru.py` | GRU risk model |
| `train_risk_iforest.py` | Isolation Forest risk model |
| `train.py` | RoBERTa LLM fine-tuning (original approach) |
| `generate_data.py` | Synthetic telemetry CSV for LLM training |

### Web Application
| File | Purpose |
|------|---------|
| `app.py` | Flask backend — API routes wrapping all core modules |
| `templates/index.html` | Single-page GUI |
| `static/style.css` | Dark theme CSS |
| `static/app.js` | Frontend logic |

### Data Files
| File | Purpose |
|------|---------|
| `tg_pfpke.db` | SQLite database (enrollments + login_attempts) |
| `keystore/` | Pickled enrollment data per user |
| `tg_risk_gru.pt` | Trained GRU weights |
| `tg_risk_iforest.pkl` | Trained Isolation Forest model |
| `SOCOFing/Real/` | 6,000 fingerprint images |

## Status
- [x] Core modules working
- [x] Three risk engines implemented and trained
- [x] CLI refactored (enroll/encrypt/decrypt subcommands)
- [ ] **IN PROGRESS**: Web GUI (Flask + HTML/CSS/JS)

## How to Run
```bash
# CLI mode
python main.py enroll "SOCOFing/Real/5__M_Left_index_finger.BMP"
python main.py encrypt "5__M_Left_index_finger" "Secret message"
python main.py decrypt "5__M_Left_index_finger" --engine gru

# Web GUI (once built)
python app.py
# Open http://localhost:5000
```

## Key Design Decisions
1. **User ID = public key** in the GUI; keystore files keyed by fingerprint filename for lookup
2. **Collapsible debug panel** at the bottom shows all intermediate values
3. **Model picker** in debug panel switches risk engine for decryption
4. **Enrollment check**: if fingerprint already enrolled, notify user instead of re-enrolling
