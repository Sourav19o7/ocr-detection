# Practical Implementation Timeline (AI-Assisted Development)

## Assumptions
- Using AI (Claude) for coding assistance
- 4-6 hours of focused development per day
- BAC team blockers resolved by end of Week 1
- Single developer + AI

---

## Detailed Task Breakdown

### Week 1: External APIs & Foundation

| Day | Tasks | Hours | Output |
|-----|-------|-------|--------|
| Day 1 | - Create external API structure<br>- Article photo upload endpoint<br>- HUID photo upload endpoint | 4-5h | Working APIs |
| Day 2 | - Auto-OCR on HUID upload<br>- File naming convention (`tagID.jpg`)<br>- Detachable item handling (`_1`, `_2`) | 4-5h | OCR integration |
| Day 3 | - Presigned URL upload flow<br>- Photo exists check endpoint<br>- API key authentication | 4h | Complete external API |
| Day 4 | - Database schema updates<br>- Add `bis_job_no`, `device_source`, `upload_status`<br>- Migration scripts | 3-4h | Updated schema |
| Day 5 | - API documentation finalization<br>- Testing external APIs<br>- Send spec to BAC team | 3h | Documentation |

**Week 1 Deliverables:**
- ✅ External APIs ready for BAC team
- ✅ API specification document sent
- ✅ Database schema updated

---

### Week 2: Excel Comparison & Queue Management

| Day | Tasks | Hours | Output |
|-----|-------|-------|--------|
| Day 1 | - Manakonline Excel parser<br>- Extract BIS Job No from filename<br>- Map columns (AHC Tag, HUID, Purity) | 4-5h | Excel parser |
| Day 2 | - Create comparison batch from Excel<br>- Match OCR HUID vs Excel HUID<br>- Match Karatage | 4-5h | Comparison logic |
| Day 3 | - Rework queue status tracking<br>- Upload queue status tracking<br>- Status transition logic | 4h | Queue management |
| Day 4 | - UI for Excel upload<br>- Comparison results view<br>- Queue views (rework/upload) | 4-5h | Frontend updates |
| Day 5 | - Testing comparison flow<br>- Bug fixes<br>- API for querying queues | 3-4h | Complete module |

**Week 2 Deliverables:**
- ✅ Manakonline Excel parsing working
- ✅ HUID comparison logic
- ✅ Rework and upload queues

---

### Week 3: Dashboard & Authentication

| Day | Tasks | Hours | Output |
|-----|-------|-------|--------|
| Day 1 | - Users table creation<br>- Password hashing (bcrypt)<br>- Login/logout endpoints | 4h | Auth backend |
| Day 2 | - Session management<br>- Role-based middleware<br>- Preset accounts (Operator/Supervisor/Admin) | 4h | Role system |
| Day 3 | - Login UI<br>- Protected routes<br>- Role-based nav | 4-5h | Auth frontend |
| Day 4 | - Global metrics aggregation<br>- Date range filter<br>- Batch filter improvements | 4h | Dashboard metrics |
| Day 5 | - CSV export endpoint<br>- Dashboard export UI<br>- Testing auth flow | 3-4h | Complete auth |

**Week 3 Deliverables:**
- ✅ User authentication working
- ✅ Role-based access control
- ✅ Enhanced dashboard with filters

---

### Week 4-5: Portal Upload Automation

| Day | Tasks | Hours | Output |
|-----|-------|-------|--------|
| W4 Day 1 | - Extension manifest setup<br>- Basic popup UI<br>- API client for backend | 4-5h | Extension skeleton |
| W4 Day 2 | - Login detection script<br>- Session state management<br>- Cookie storage | 4-5h | Session handling |
| W4 Day 3 | - Upload queue fetch from API<br>- Image download from S3<br>- Basic navigation | 4-5h | Queue integration |
| W4 Day 4 | - DOM automation: find Job No<br>- DOM automation: find Tag row<br>- DOM automation: click Browse | 5-6h | Portal navigation |
| W4 Day 5 | - File upload injection<br>- Upload confirmation detection<br>- Status reporting | 5-6h | Upload automation |
| W5 Day 1 | - Error handling<br>- Retry logic<br>- Session expiry detection | 4h | Resilience |
| W5 Day 2 | - Progress UI in popup<br>- Status bar overlay<br>- Pause/resume functionality | 4-5h | Enhanced UI |
| W5 Day 3 | - Backend: upload queue API<br>- Backend: result reporting API<br>- Backend: statistics API | 4h | Backend support |
| W5 Day 4 | - Integration testing<br>- Portal-specific fixes<br>- Performance tuning | 4-5h | Testing |
| W5 Day 5 | - Documentation<br>- Operator guide<br>- Edge case handling | 3-4h | Documentation |

**Week 4-5 Deliverables:**
- ✅ Browser extension working
- ✅ Semi-automated upload flow
- ✅ Error handling & retry

---

### Week 6: Testing & Deployment

| Day | Tasks | Hours | Output |
|-----|-------|-------|--------|
| Day 1 | - End-to-end testing (full flow)<br>- Test with real data | 4-5h | Test results |
| Day 2 | - Bug fixes from testing<br>- Performance optimization | 4-5h | Fixes |
| Day 3 | - Operator documentation<br>- Training material prep | 3-4h | Documentation |
| Day 4 | - Deployment setup<br>- Production configuration | 3-4h | Deployment ready |
| Day 5 | - Operator training<br>- Go-live support | 4h | Live system |

**Week 6 Deliverables:**
- ✅ System tested and deployed
- ✅ Operators trained
- ✅ System live

---

## Risk Buffer

Built-in buffer for:
- Unexpected bugs: 2-3 days
- Portal changes: 1-2 days
- BAC integration issues: 2-3 days

---

## Parallel Tracks

Some work can happen in parallel:

```
Week 1: [External APIs] ─────────────────────────────▶
        [BAC team integration] ─────────────────────▶ (their side)

Week 2: [Excel Comparison] ──────────────────────────▶
        [BAC camera integration] ───────────────────▶ (their side)

Week 3: [Dashboard & Auth] ──────────────────────────▶

Week 4-5: [Portal Automation] ───────────────────────▶

Week 6: [Testing & Deployment] ──────────────────────▶
```

---

## Summary

| Phase | Duration | Key Dependency |
|-------|----------|----------------|
| External APIs | Week 1 | None |
| Excel Comparison | Week 2 | Excel format confirmed |
| Dashboard & Auth | Week 3 | None |
| Portal Automation | Week 4-5 | Portal credentials |
| Testing & Deploy | Week 6 | All features complete |

**Total: 6 weeks** to full deployment

**With blockers unresolved:** Add 1-2 weeks depending on BAC team response time
