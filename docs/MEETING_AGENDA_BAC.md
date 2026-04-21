# BAC Team Meeting Agenda - Hallmark QC Integration

## Date: [Tomorrow's Date]
## Attendees: QC Team, BAC Team

---

## Meeting Objective

Align on integration requirements, resolve blockers, and establish timeline for Hallmark QC system implementation.

---

## Agenda Items

### 1. Critical Blocker: ERP APIs Not Working (10 min)

**Issue:**
- ERP APIs provided in Postman collection are not accessible
- Base URL: `http://103.133.214.232:8026`
- All endpoints returning errors/timeouts

**APIs Affected:**
| API | Purpose | Status |
|-----|---------|--------|
| `/api/grn/Receipt` | Get receipt info | NOT WORKING |
| `/api/grn/GRN` | Get GRN details | NOT WORKING |
| `/api/grn/HUID` | Get HUID data | NOT WORKING |
| `/api/grn/HUID_API` | Get HUID for tag | NOT WORKING |

**Required Action:**
- [ ] BAC team to verify API server status
- [ ] Provide working base URL
- [ ] Share sample API responses for each endpoint

**Impact if not resolved:** Cannot proceed with integration

---

### 2. Camera Integration Approach (15 min)

**Context:**
- BAC team manages camera hardware (article camera, microscope)
- We need images sent to our system for processing

**Proposed Solution:**
We provide APIs, BAC team integrates camera software to call our APIs.

**APIs We Will Provide:**

| API | Purpose |
|-----|---------|
| `POST /api/external/article-photo` | Upload article image |
| `POST /api/external/huid-photo` | Upload HUID image (triggers OCR) |
| `GET /api/external/photo-exists/{tag_id}` | Check if photos exist |

**Questions for BAC Team:**
1. Can camera software make HTTP POST requests?
2. What format do cameras output? (JPEG, PNG, RAW?)
3. Can camera software handle multipart/form-data upload?
4. Alternative: Can cameras save to a shared folder we can monitor?

**Decision Needed:**
- [ ] Confirm integration approach (API upload vs. folder monitoring)
- [ ] Assign BAC team member for camera integration

---

### 3. ERP Data Requirements (10 min)

**What we need from ERP APIs:**

| Data Point | From API | Purpose |
|------------|----------|---------|
| Tag ID | HUID_API | Item identification |
| BIS Job No | GRN | Portal matching |
| Expected HUID | HUID_API | OCR comparison |
| Declared Purity | GRN | Karatage verification |
| Detachable flag | Tag ID suffix | Identify parts |

**Questions:**
1. Does `HUID_API` return the assigned HUID for a tag?
2. What is the response structure of each API?
3. Can we get sample JSON responses?

**Required Action:**
- [ ] BAC team to share API response samples
- [ ] Confirm data availability for all required fields

---

### 4. BIS Export Format (5 min)

**Context:**
- Need to export data in BIS-required format for compliance
- Format template not yet available

**Required Action:**
- [ ] BAC team to provide BIS export template/sample
- [ ] Clarify required columns and format specifications

**Timeline:** Need by Week 3

---

### 5. Portal Automation Coordination (10 min)

**Context:**
- Portal has CAPTCHA (cannot fully automate)
- Proposing semi-automated browser extension

**How it works:**
1. Operator logs in manually (solves CAPTCHA)
2. Extension automates repetitive uploads
3. Handles 1000+ uploads/day

**Questions:**
1. Can we get test portal credentials for development?
2. Are there any IP restrictions on portal access?
3. Any known rate limits on uploads?
4. Is portal UI expected to change soon?

**Required Action:**
- [ ] Provide UAT portal test credentials
- [ ] Confirm no restrictions on automated uploads

---

### 6. Timeline & Milestones (10 min)

**Proposed Timeline:**

| Week | Milestone | Dependency |
|------|-----------|------------|
| 1 | External APIs ready | API spec approval |
| 2 | Excel comparison working | Manakonline Excel format |
| 3 | Dashboard & basic auth | None |
| 4-5 | Portal extension | Portal credentials |
| 6 | Testing & go-live | All blockers resolved |

**Key Dependencies on BAC Team:**

| Item | Needed By | Priority |
|------|-----------|----------|
| Working ERP APIs | Week 1 | CRITICAL |
| Sample API responses | Week 1 | CRITICAL |
| Camera integration owner | Week 1 | HIGH |
| Portal test credentials | Week 3 | HIGH |
| BIS export template | Week 3 | MEDIUM |

---

## Summary of Action Items

### For BAC Team:
1. [ ] Fix ERP API accessibility
2. [ ] Share sample API responses
3. [ ] Assign camera integration owner
4. [ ] Provide portal test credentials
5. [ ] Share BIS export template

### For QC Team:
1. [ ] Finalize API specification
2. [ ] Set up S3 bucket with proper permissions
3. [ ] Begin development of external APIs
4. [ ] Document integration guide

---

## Risks to Highlight

| Risk | Impact | Mitigation |
|------|--------|------------|
| ERP APIs remain inaccessible | Project blocked | Escalate immediately |
| Camera integration delays | Module 1 & 2 delayed | Start portal work first |
| Portal UI changes | Extension breaks | Build flexible selectors |

---

## Next Meeting

- **Date:** [One week from now]
- **Agenda:** Integration progress review, demo of APIs

---

## Contact Points

| Area | BAC Team | QC Team |
|------|----------|---------|
| ERP APIs | [Name TBD] | [Your Name] |
| Camera Integration | [Name TBD] | [Your Name] |
| Portal Automation | [Name TBD] | [Your Name] |
