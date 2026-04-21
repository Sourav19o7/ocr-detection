# Email Draft: Hallmark QC Integration - Roadblocks & Discussion Points

---

**To:** BAC Team
**Cc:** [Stakeholders]
**Subject:** Hallmark QC Integration - Critical Blockers & Meeting Request

---

Hi Team,

We have completed the analysis for the Hallmark QC System implementation covering all 5 modules (Article Photo Capture, HUID Photo Capture, OCR Verification, Portal Upload Automation, and Dashboard).

Before we proceed with development, we need to resolve some **critical blockers** and align on the integration approach. I'm requesting a meeting to discuss these points.

---

## Critical Blockers

### 1. ERP APIs Not Accessible

The ERP APIs provided in the Postman collection are currently not working:

- **Base URL:** `http://103.133.214.232:8026`
- **Affected APIs:** Receipt, GRN, Sampling, Fireassay, Dispatch, HUID, QC, XRF, HUID_API
- **Issue:** Connection timeout / not reachable

**Impact:** We cannot integrate with ERP data without working APIs. This blocks the entire Tag ID and HUID validation workflow.

**Request:** Please verify server status and provide working endpoints.

---

### 2. Sample API Responses Needed

Even after APIs are fixed, we need sample JSON responses to design our data mapping correctly.

**Required samples for:**
- `/api/grn/Receipt`
- `/api/grn/GRN`
- `/api/grn/HUID`
- `/api/grn/HUID_API` (with Tag ID parameter)

---

### 3. Camera Integration Approach

Since ERP team will not implement POST APIs, we propose:

**Our team provides APIs → Your camera software calls our APIs**

We have prepared the API specification document (attached). Please confirm:
- Can camera software make HTTP POST requests?
- Who will be the integration owner from BAC team?

---

### 4. BIS Export Template

We need the BIS-required export format template to implement the export functionality. Please share the template or sample file.

---

### 5. Portal Test Credentials

For developing the semi-automated upload tool, we need:
- UAT portal access credentials
- Confirmation of any IP restrictions or rate limits

---

## Discussion Points for Meeting

1. ERP API status and timeline to fix
2. Camera integration approach sign-off
3. Data mapping review (what fields are available from ERP)
4. Portal automation coordination
5. Overall timeline alignment

---

## Proposed Timeline (Pending Blocker Resolution)

| Week | Deliverable |
|------|-------------|
| 1 | External APIs ready for camera integration |
| 2 | Manakonline Excel comparison working |
| 3 | Dashboard & user authentication |
| 4-5 | Portal upload browser extension |
| 6 | Testing & go-live |

**Total: 6 weeks** (assuming blockers resolved by Week 1)

---

## Attachments

1. `API_SPECIFICATION_FOR_BAC.md` - API specs for camera integration
2. `PORTAL_AUTOMATION_OVERVIEW.md` - How the upload tool will work
3. `IMPLEMENTATION_ROADMAP.md` - Full implementation plan

---

Please confirm a time for the meeting (30-45 minutes).

Best regards,
[Your Name]

---

## Quick Reference: What We Need from BAC Team

| # | Item | Priority | Needed By |
|---|------|----------|-----------|
| 1 | Working ERP APIs | CRITICAL | ASAP |
| 2 | Sample API responses | CRITICAL | Before meeting |
| 3 | Camera integration owner assigned | HIGH | Week 1 |
| 4 | Portal test credentials | HIGH | Week 3 |
| 5 | BIS export template | MEDIUM | Week 3 |
