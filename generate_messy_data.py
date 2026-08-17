"""
Data Generator: Creates messy raw corporate data for the compliance pipeline.

Generates:
  data/raw_inputs/messy_ledger.csv         -- Large dirty ERP transaction ledger (35+ rows)
  data/raw_inputs/invoice_*.txt            -- 8 messy invoice text files (OCR-style noise)
  data/documents/*.txt                     -- 22 dense policy documents for RAG (plain text)

Usage:
  python generate_messy_data.py
"""
import glob
import os
from pathlib import Path


def create_directories():
    """Create the necessary folder structure."""
    dirs = ["data/raw_inputs", "data/documents", "data/faiss_index", "output"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}/")


def cleanup_legacy_files():
    """Remove leftover files from older generator versions."""
    removed = 0
    for f in glob.glob("data/documents/*.md") + glob.glob("data/raw_inputs/ocr_*.txt"):
        os.remove(f)
        removed += 1
    if removed:
        print(f"  Removed {removed} legacy files")


def generate_messy_csv():
    """
    Generate a large, realistically messy CSV ledger with 35+ entries.
    Problems deliberately introduced:
      - 8 different date formats mixed across rows
      - Missing Ref_IDs on several rows
      - Vendor names: mixed case, extra spaces, symbols, underscores
      - Amount column: trailing spaces, missing values, inconsistent sign
      - Status column: mixed case (PENDING, cleared, Cleared, CLEARED, pending)
      - Missing Dept_Code on several rows
      - Duplicate vendor entries (to test duplicate-payment detection)
      - Various department codes: ENG-01, SLS-99, HR-03, FIN-04, MKT-05
    """
    headers = "Txn_Date,Ref_ID,Vendor_Desc,Amt,Status,Notes,Dept_Code\n"
    rows = [
        # --- Cloud Infrastructure (ENG-01) ---
        "2026-08-01,TX-9901,AWS* CLOUD SERVICES,-4500.00,PENDING,monthly EC2 compute,ENG-01\n",
        "08/03/26,,AMAZON WEB SERV  ,-800.00,cleared,S3 storage charges,ENG-01\n",
        "2026-08-05,TX-9903,  Cloudflare Inc ,  -8450.00, PENDING, enterprise CDN annual,ENG-01\n",
        "08-06-2026,TX-9904,cloudflare inc,-320.00,CLEARED,overage - bandwidth,ENG-01\n",
        "2026.08.07,TX-9905,GOOGLE CLOUD PLATFORM,-3100.00,PENDING,GKE cluster monthly,ENG-01\n",
        "Aug 08 2026,TX-9906,  Azure Microsoft ,-2750.00,pending,dev environment,ENG-01\n",
        "26-08-09,TX-9907,AWS* CLOUD SERVICES,-5200.00,PENDING,overage fees August,ENG-01\n",
        "08/10/26,TX-9908,  DigitalOcean  LLC,-450.00,CLEARED,droplets - staging,ENG-01\n",
        "2026-08-11,,amazon web serv,-1100.00,Cleared,CloudFront distribution,\n",
        "2026-08-12,TX-9910,CLOUDFLARE  INC  ,-220.00,CLEARED,additional IP addresses,ENG-01\n",

        # --- SaaS / Dev Tools (ENG-01) ---
        "08/13/26,,slack technologies inc,-1200.00,cleared,,\n",
        "2026-08-14,TX-9912,DATADOG   INC,-3400.00,PENDING,APM + Logs Pro plan,ENG-01\n",
        "2026.08.15,tx-9913,GitHub_Enterprise,-850.00,Cleared,50 developer seats,ENG-01\n",
        "08-17-2026,TX-9914,  Datadog Inc , -3400.00 , PENDING ,,ENG-01\n",
        "Aug 18 2026,TX-9915,GITHUB  ENTERPRISE,-850.00,cleared,renewal - annual,ENG-01\n",
        "26-08-19,TX-9916,jira-confluence atlassian,-2100.00,PENDING,team subscription,ENG-01\n",
        "2026-08-20,TX-9917,  ZOOM  VIDEO COMMUNICATIONS  ,-1800.00,CLEARED,enterprise plan,\n",
        "08/21/26,,figma inc,-600.00,cleared,design team seats,ENG-01\n",
        "2026-08-22,TX-9919,PagerDuty,-490.00,CLEARED,incident management,ENG-01\n",
        "2026.08.23,TX-9920,  Notion Labs Inc ,-380.00,Cleared,team workspace annual,ENG-01\n",

        # --- Sales & Travel (SLS-99) ---
        "26-08-05,TX-9925,UBER   TRVL,-45.50,CLEARED,sales trip - client dinner,SLS-99\n",
        "08-08-2026,TX-9930,  Delta airlines  ,-620.00,CLEARED,NYC client visit,SLS-99\n",
        "2026-08-10,TX-9931,MARRIOTT  HOTELS,-310.00,cleared,NYC hotel 2 nights,SLS-99\n",
        "Aug 12 2026,,lyft business  ,-38.00,CLEARED,airport transfer,SLS-99\n",
        "2026.08.14,TX-9933,Restaurant Le Bernardin,-285.00,CLEARED,client entertainment dinner,SLS-99\n",
        "08/16/26,TX-9934,  UNITED AIRLINES,-890.00,PENDING,SF conference travel,SLS-99\n",
        "2026-08-18,TX-9935,hilton garden inn,-195.00,cleared,SF hotel 1 night,SLS-99\n",
        "26-08-20,TX-9936,EXPENSIFY CORP,-99.00,CLEARED,expense management tool,SLS-99\n",

        # --- HR & Recruiting (HR-03) ---
        "2026-08-04,TX-9940,  LinkedIn Recruiter ,-1200.00,PENDING,monthly recruiting seats,HR-03\n",
        "08/09/26,,workday inc  ,-4200.00,cleared,HRIS monthly subscription,HR-03\n",
        "2026-08-13,TX-9942,BAMBOO HR,-780.00,CLEARED,HR platform - 100 employees,HR-03\n",
        "2026.08.22,TX-9943,  Indeed Job Postings,-540.00,Cleared,sponsored job listings,HR-03\n",

        # --- Finance / Legal (FIN-04) ---
        "08-02-2026,TX-9950,QUICKBOOKS  ONLINE,-300.00,CLEARED,accounting software,FIN-04\n",
        "Aug 05 2026,,Stripe Inc,-125.00,cleared,payment processing fees,FIN-04\n",
        "2026-08-15,TX-9952,  DocuSign Enterprise  ,-960.00,PENDING,e-signature annual plan,FIN-04\n",

        # --- Marketing (MKT-05) ---
        "26-08-08,TX-9960,HUBSPOT INC,-2400.00,PENDING,marketing CRM monthly,MKT-05\n",
        "08/17/26,,semrush pro,-450.00,cleared,SEO tool annual,MKT-05\n",
        "2026-08-24,TX-9962,  GOOGLE  ADS  , -3800.00, PENDING, August campaign budget,MKT-05\n",

        # --- Suspicious / Edge-case rows ---
        "2026-08-20,TX-9921,AWS* CLOUD SERVICES,-4500.00,PENDING,monthly hosting,ENG-01\n",
        "2026-08-28,,  UNKNOWN VENDOR  ,-12500.00,PENDING,,\n",
        "08/30/26,TX-9999,  , , ,no vendor info - investigate,,\n",
    ]

    filepath = Path("data/raw_inputs/messy_ledger.csv")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(headers)
        f.writelines(rows)
    print(f"  Generated: {filepath}  ({len(rows)} data rows)")


def generate_invoices():
    """
    Generate 8 messy invoice text files. These are not OCR output — they are
    hand-written samples that mimic OCR-style typos and formatting quirks.
    """
    invoices = {

        "invoice_cloudflare_9982.txt": """\
INVOICE #9982-A
DAT: 08-10-2026
Vndor: Cloudflare, Inc.
---
BILLED TO:
Acme Corp Accounts Payable
ATTN: Network Infrastructure
---
Total Amnt Due: $ 8,450.00
Tax: 0.00
---
Notes: Enterprise plan annual. Please remit payment within Net30 terms. Late fees apply.""",

        "invoice_aws_88219.txt": """\
== INVOICE ==
No: INV-AWS-88219
Date::: 08/25/26

To: ACME Corp (Dept: ENG-01)
From: Amazon Web Services (AWS)

Description                  Amount
-----------------------------------
EC2 Instances (us-east-1)    $ 4200.50
S3 Standard Storage          $  800.00
Data Transfer Out            $  199.50

TOTAL DUE NOW:::: $ 5200.00
Terms: Due upon receipt.""",

        "invoice_datadog_99341.txt": """\
-- Datadog, Inc. --
Invoice Number : INV-DDG-99341
Issued  : 08-15-2026

Bill To  : Acme Corp // Engineering Dept (ENG-01)

SERVICES RENDERED:
  APM Pro Plan (50 hosts)         USD 2,800.00
  Log Management (500 GB/day)     USD   400.00
  Synthetics Monitoring           USD   200.00
  --------------------------------
  Sub-total                       USD 3,400.00
  Tax (0%)                        USD     0.00
  AMOUNT OWING:                   USD 3,400.00

Payment Terms: Net-30 from invoice date.
Please reference PO# PO-2026-0088 on your payment.""",

        "invoice_github_enterprise_4421.txt": """\
GITHU8 ENTERPRISE INVOICE
[OCR may have misread some characters]

Inv No:  GH-ENT-4421
Dat e:   Aug 14  2026

Custr:   Acme Corp
         Engineering Division

License: GitHub Enterprise Cloud
         50 seats x $17.00/seat/month

Ttal Amt:   $ 850 .00
Paymnt Trms: NET 30

Wire Transfer Details:
Bank: Silicon Valley Bank
Account: [REDACTED]

Note: This is a duplicate renewal - original invoice GH-ENT-4415 was issued Aug 1 2026.""",

        "invoice_zoom_enterprise_7751.txt": """\
Z O O M  VIDEO  COMMUNICATIONS
- - - - - - - - - - - - - - -
Invoice:  ZM-7751-ENT
Date:     2O26-O8-17      [OCR note: zeros may appear as O]

Sold To:
  Acme Corp
  Attn: IT Procurement

Plan: Enterprise Business (500 hosts)
Term: Annual

Amount:   $ 1 , 8 0 0 . 0 0
Tax:      0.00
TOTAL:    $ 1,800.00

Payment: Due Upon Receipt
Support: enterprise-billing@zoom.us""",

        "invoice_google_cloud_55018.txt": """\
Google Cloud Platform
Monthly Usage Statement

Account ID:  acme-corp-prod-001
Billing Period:  Aug 01 - Aug 31, 2026
Invoice #:  GCP-55018

Service                      Usage           Cost
-----------------------------------------------------
Compute Engine (n2-standard)  2,400 hrs       $1,840.00
Cloud Storage (Standard)      50 TB           $  500.00
BigQuery (on-demand)          12 TB scanned   $  600.00
Cloud SQL (db-n1-standard-4)  744 hrs         $  260.00
-----------------------------------------------------
SUBTOTAL                                      $3,200.00
Discount (committed use)                      $ -100.00
TAX                                           $    0.00
AMOUNT DUE:                                   $3,100.00

Payment Terms: Charged to credit card on file.
Dept Code: ENG-01
This charge will process on 2026-09-05.""",

        "invoice_slack_enterprise_3302.txt": """\
INVOICE
Slack Technologies, Inc.

Inv.# : SL-ENT-3302
Date  : 08/14/26

TO:
Acme Corp - IT Department
Accounts Payable

DESCRIPTION
Slack Enterprise Grid
  Workspace: acme.slack.com
  Active members: 240
  Rate: $5.00/member/month

Monthly Total: 240 x $5.00 = $1,200.00

TOTAL DUE: $1,200.00

Note: Annual commitment discount of 15% applies if converted to annual plan.
Terms: Net-3O  [possible OCR error - intended Net-30]""",

        "invoice_microsoft_azure_9010.txt": """\
Microsoft Azure
Cloud Services Invoice

Invice Number: AZ-9O1O-2026    <- OCR artifact, actual: AZ-9010-2026
Billing Cycle: Aug 1 - Aug 31, 2026

Customer: Acme Corporation
Dept: Engneerng Dept ENG-01    <- OCR artifact

Services:
  Virtual Machines (B-series)   $1,200.00
  Azure Blob Storage            $  350.00
  Azure Active Directory P2     $  800.00
  Azure DevOps (50 users)       $  400.00
  -----------------------------------------
  Total Before Tax:             $2,750.00
  Tax:                          $    0.00
  -----------------------------------------
  AMOUNT DUE:                   $2,750.00

Payment Due: Remit within 30 days of invoice date (Net30)
PO Reference: PO-AZ-2026-007""",
    }

    for filename, content in invoices.items():
        with open(f"data/raw_inputs/{filename}", "w", encoding="utf-8") as f:
            f.write(content)

    print(f"  Generated: {len(invoices)} invoice text files in data/raw_inputs/")


def generate_policy_documents():
    """
    Generate 22 dense corporate policy documents as plain .txt files.
    These form the RAG knowledge base that the Retriever agent queries.
    """
    docs = {
        "corporate_procurement_policy.txt": """\
Acme Corp Global Procurement and Financial Policy
Document Version: 4.1 | Effective Date: Jan 1, 2026

Section 4.2.1: Cloud Infrastructure Procurement
Any expenses related to cloud hosting, including but not limited to Amazon Web Services (AWS), Google Cloud Platform, Microsoft Azure, and Cloudflare, must be categorized under the ENG-01 department code. Monthly aggregate spending exceeding $5,000.00 for any single cloud vendor requires secondary authorization from the VP of Engineering before invoices are processed. Furthermore, all vendor invoices exceeding $2,000.00 must adhere to Net-30 payment terms; invoices requesting immediate payment (Due Upon Receipt or Due Now) will be flagged for manual review and require vendor renegotiation before funds are released.

Section 4.2.2: Software as a Service (SaaS) Subscriptions
Standard developer tooling such as GitHub Enterprise, Slack, Datadog, Zoom, and similar platforms falls under the ENG-01 operational budget. Individual subscriptions exceeding $1,000.00 annually must be consolidated into enterprise agreements. Duplicate payments to the same vendor within a 30-day window will be automatically rejected unless accompanied by an approved variance report signed by the department director. GitHub Enterprise renewals must reference the original purchase order number.

Section 4.2.3: Travel and Entertainment Expenses
All travel-related expenses must be categorized under SLS-99 (Sales) or the relevant department code. Individual meal and entertainment expenses exceeding $150.00 per person require manager approval. International travel exceeding $3,000.00 requires C-level sign-off at least 5 business days before departure.

Section 4.2.4: Unknown or Unverified Vendors
Any payment to a vendor not registered in the Approved Vendor Registry (AVR) must be blocked and escalated. Payments to unnamed or unidentifiable vendors are prohibited without CFO written approval. Transactions marked as UNKNOWN VENDOR require immediate investigation by the Internal Audit team.
""",
        "vendor_management_policy.txt": """\
Vendor Management and Approval Policy
Document: VMP-2026-01 | Owner: Procurement Office

Approved Vendor Registry
All vendors receiving payments must be registered in the Acme Corp Approved Vendor Registry (AVR) before any purchase order is issued. Vendors not in the AVR will have payments withheld until registration is complete. The AVR review cycle occurs quarterly. Cloud vendors currently approved include: Amazon Web Services (AWS), Google Cloud Platform, Microsoft Azure, Cloudflare, DigitalOcean, and Fastly.

Vendor Due Diligence
New vendors requesting contracts above $10,000.00 annually must undergo a security and financial due diligence review. This includes SOC 2 Type II certification for any vendor handling data classified as Confidential or above. Vendors must re-certify annually.

Payment Terms Negotiation
The preferred payment terms for all vendors are Net-30 from invoice date. Exceptions to Net-30 terms such as prepayments or immediate payment require written approval from the CFO and must be documented in the vendor contract. Vendors offering early-payment discounts such as 2/10 Net-30 should be evaluated against the company's cost of capital.
""",
        "accounts_payable_procedures.txt": """\
Accounts Payable Standard Operating Procedures
Document: AP-SOP-2026 | Department: Finance

Invoice Processing Requirements
All invoices must include: vendor name, invoice number, invoice date, itemized line items, total amount, and payment terms. Invoices missing any required field will be returned to the vendor unpaid. OCR-processed invoices from PDF sources must be validated against the original document by an AP specialist before payment.

Three-Way Match Requirement
All purchases above $500.00 require a three-way match: Purchase Order (PO), Goods Receipt Note (GRN), and vendor invoice. Payments without a valid three-way match will be blocked in the ERP system.

Cloud Vendor Invoice Handling
Cloud vendor invoices from AWS, Cloudflare, GCP, and Azure are processed monthly. Any single vendor invoice or aggregate monthly total exceeding $5,000 triggers an automatic hold pending VP of Engineering authorization per Section 4.2.1 of the Procurement Policy. The authorization must be received within 5 business days or the invoice is returned to the vendor. Invoices marked Due Upon Receipt or Due Now from these vendors are automatically flagged as policy violations.

Duplicate Invoice Detection
The AP system automatically flags duplicate invoices from the same vendor within any 30-day rolling window. Duplicate amounts must be manually verified. GitHub Enterprise invoices GH-ENT-4415 and GH-ENT-4421 represent a known duplicate scenario requiring investigation.
""",
        "travel_expense_policy.txt": """\
Corporate Travel and Expense Policy
Document: TEP-2026 | Effective: Jan 1, 2026

Domestic Travel
Domestic flights must be booked in Economy class unless the flight exceeds 4 hours, in which case Premium Economy is permitted. Hotel accommodations are capped at $250 per night for major metropolitan areas and $180 per night for secondary markets. Meal per diem is $75 per day for domestic travel.

Client Entertainment
Client entertainment expenses such as meals and events are reimbursable up to $150 per person per event. Events exceeding $500 total must be pre-approved by the department VP. All entertainment expenses must include a business justification, list of attendees, and client or prospect name.

Department Codes for Travel
Sales-related travel must be coded to SLS-99. Engineering conference travel is coded to ENG-01. All travel must be booked through the corporate travel portal to qualify for reimbursement.
""",
        "saas_procurement_guidelines.txt": """\
SaaS Procurement and License Management Guidelines
Document: SAAS-2026-03 | Owner: IT and Procurement

Approved SaaS Platforms
The following SaaS tools are centrally managed under enterprise agreements: Slack (communication), GitHub Enterprise (source control), Datadog (monitoring), Zoom (video conferencing), Salesforce (CRM), Jira and Confluence (project management), HubSpot (marketing CRM), Workday (HRIS). Individual departments may not procure shadow SaaS tools without IT approval.

License Tier Thresholds
Individual SaaS tool licenses costing more than $1,000.00 per year must be reviewed and approved by IT Procurement before purchase. Tools costing more than $5,000.00 per year require a vendor evaluation committee review.

Renewal Management
IT maintains a SaaS renewal calendar. Departments must notify IT 60 days before any SaaS contract renewal. Unauthorized auto-renewals that were not flagged in the renewal calendar will be disputed. Zoom Enterprise is flagged as requiring renewal review in August 2026. Slack Enterprise Grid for 240 members at $5.00 per member per month is within the approved budget envelope.
""",
        "financial_controls_framework.txt": """\
Financial Controls and Internal Governance Framework
Document: FC-2026 | Owner: CFO Office

Delegation of Authority Matrix
Purchases up to $1,000: Manager approval required.
Purchases $1,001 to $5,000: Director-level approval required.
Purchases $5,001 to $25,000: VP-level approval required, including secondary authorization from VP of Engineering for all technology and cloud infrastructure purchases.
Purchases above $25,000: CFO approval required with Board notification.
All cloud infrastructure spending is subject to the thresholds defined in Section 4.2.1 of the Procurement Policy regardless of contract vehicle.

Segregation of Duties
No single employee may initiate, approve, and process a payment. The following controls are enforced: purchase requisition raised by requester, purchase order approved by separate manager, invoice matched and approved by AP team, payment released by treasury.

Anomaly Detection Rules
The finance system flags: duplicate invoices from the same vendor within 30 days, invoices with Due Upon Receipt terms above $2,000, cloud spend exceeding $5,000 monthly per vendor (Cloudflare $8,450 is a clear violation of this rule), missing department codes, unmatched purchase orders, and any transaction to an unknown or unregistered vendor.
""",
        "cloud_security_standards.txt": """\
Cloud Infrastructure Security Standards
Document: CSS-2026-02 | Owner: Information Security

Cloud Provider Security Requirements
All cloud providers used by Acme Corp must maintain at minimum: ISO 27001 certification, SOC 2 Type II reports updated annually, GDPR compliance documentation, and 99.9 percent SLA uptime guarantees with credits for downtime. Approved providers include AWS, GCP, Azure, Cloudflare, and DigitalOcean. All cloud workloads must be deployed in regions compliant with Acme Corp data residency requirements.

Cost Governance in Cloud
Cloud cost governance is a shared responsibility between Engineering and Finance. Monthly cloud spend reports must be submitted to the VP of Engineering and CFO by the 3rd business day of each month. Budget overruns of more than 20 percent above the approved monthly plan must be escalated to the CFO within 24 hours.

Resource Tagging
All cloud resources must be tagged with mandatory tags: Department such as ENG-01, Project, Environment such as prod or staging or dev, CostCenter, and Owner. Untagged resources running for more than 7 days will be automatically stopped pending tag compliance.
""",
        "budget_authorization_matrix.txt": """\
Budget Authorization and Spending Matrix
Document: BAM-2026 | Owner: Finance and Legal

Technology Budget Authorization
Engineering department ENG-01 has an approved annual budget allocation. Monthly cloud infrastructure spend above $5,000 USD for any single vendor requires written sign-off from the VP of Engineering before invoices are processed. This authorization must reference the relevant purchase order number and be submitted to accounts payable within 2 business days of invoice receipt.

Specific Vendor Thresholds
AWS monthly aggregate spending above $5,000 triggers VP Engineering authorization. This applies when AWS EC2 overage fees of $5,200 are combined with regular hosting charges. Cloudflare annual invoice of $8,450 clearly exceeds the $5,000 monthly authorization threshold and requires both VP Engineering sign-off and CFO review given the aggregate amount. Google Cloud monthly invoice of $3,100 is within director-level approval range. Datadog invoices at $3,400 each require director approval and duplicate payment investigation.

Emergency Authorization
In cases where critical infrastructure requires immediate payment, an expedited approval process is available. The requestor must submit a Critical Service Authorization form and obtain verbal approval from the VP of Engineering followed by written confirmation within 24 hours. This process does not waive the Net-30 payment terms requirement.
""",
        "payment_processing_standards.txt": """\
Payment Processing and Terms Standards
Document: PPS-2026 | Owner: Treasury and Finance

Standard Payment Terms
Acme Corp standard payment terms are Net-30 from the invoice date. This means payment will be initiated 30 days after the invoice date. All vendor agreements must specify Net-30 terms unless a specific exception has been approved by the CFO in writing. Any invoice presenting Due Upon Receipt, Due on Delivery, Immediate Payment Required, or Due Now terms will be automatically flagged by the AP system for manual review and possible rejection.

Problematic Payment Term Examples
The AWS invoice INV-AWS-88219 with terms Due Upon Receipt for $5,200 is a double violation: it exceeds the $5,000 monthly authorization limit AND uses non-standard payment terms. Zoom invoice ZM-7751-ENT marked Due Upon Receipt for $1,800 violates Net-30 policy for invoices exceeding $2,000 and must be renegotiated before payment.

Vendor Invoice Requirements
For an invoice to be processed, it must be addressed to Acme Corp with correct entity name, include a unique invoice number, reference the purchase order number if applicable, show itemized line items with clear descriptions, display correct payment terms Net-30, and be received within 90 days of service delivery.
""",
        "software_license_compliance.txt": """\
Software License Compliance Policy
Document: SLC-2026 | Owner: Legal and IT

License Audit Requirements
Acme Corp conducts annual software license audits for all deployed software. Departments are responsible for maintaining accurate records of all software installations and license counts. Discrepancies between licensed seats and actual installations must be remediated within 30 days of audit findings.

Enterprise Agreement Compliance for SaaS
GitHub Enterprise, Slack, and Datadog are covered under enterprise agreements negotiated by IT Procurement. Departments must not purchase additional seat licenses outside these agreements. The GitHub Enterprise duplicate invoices GH-ENT-4415 and GH-ENT-4421 issued within the same month represent a potential duplicate payment violation requiring immediate investigation by AP and Internal Audit. Individual seat purchases for approved tools outside the enterprise agreement will be rejected.

Open Source Usage Policy
Open source software used in production systems must be reviewed for license compatibility. GPL-licensed software may not be incorporated into proprietary products without Legal approval. MIT, Apache 2.0, and BSD-licensed software is generally approved for use.
""",
        "internal_audit_framework.txt": """\
Internal Audit and Compliance Monitoring Framework
Document: IAF-2026 | Owner: Internal Audit

Continuous Transaction Monitoring
The internal audit function maintains a continuous transaction monitoring program reviewing all vendor payments above $1,000. Automated rules applied include: duplicate payment detection within 30 days, vendor not in approved registry check, payment terms compliance check for invoices above $2,000, department code validation, monthly aggregate spend limit enforcement per Section 4.2.1, and detection of any transaction to an unidentified or blank vendor.

August 2026 Compliance Alerts
The following items from the August 2026 transaction register require immediate investigation: (1) Cloudflare $8,450 single invoice exceeds $5,000 authorization threshold, (2) AWS $5,200 overage with Due Upon Receipt payment terms is a dual violation, (3) Two Datadog invoices for $3,400 each within the same period represent a potential duplicate, (4) Two GitHub Enterprise invoices within one month require duplicate payment review, (5) Unknown vendor transaction for $12,500 with no department code is flagged as critical, (6) Zoom $1,800 with Due Upon Receipt terms requires renegotiation.

Audit Findings Classification
Critical findings require immediate executive notification and remediation within 5 business days. High findings require remediation within 30 days. All findings are tracked in the audit management system.
""",
        "risk_management_framework.txt": """\
Enterprise Risk Management Framework
Document: ERMF-2026 | Owner: Risk Committee

Financial Risk Categories
Acme Corp classifies financial risks into four categories: Credit Risk which is the risk of vendor default or non-delivery after payment, Operational Risk which is the risk of control failures in procurement and payment processes, Compliance Risk which is the risk of violating regulatory requirements or internal policies, and Concentration Risk which is the risk of over-reliance on single vendors particularly for critical cloud infrastructure.

Vendor Concentration Limits
No single cloud vendor should represent more than 60 percent of total cloud spend. Current policy requires diversification across at least two cloud providers for mission-critical services. The monthly AWS spend report is reviewed by the VP of Engineering and Risk Committee to monitor concentration. Cloud spend exceeding $5,000 monthly per vendor is a trigger event requiring VP authorization.

Control Failure Escalation
Control failures detected by the AP system are escalated as follows: financial impact under $5,000 goes to department manager, $5,000 to $25,000 goes to VP level, above $25,000 goes to CFO and Legal. All escalations must be documented and closed within the specified remediation window.
""",
        "supplier_code_of_conduct.txt": """\
Supplier Code of Conduct
Document: SCC-2026 | Owner: Procurement and Legal

Ethical Standards for Vendors
All vendors supplying goods or services to Acme Corp must adhere to the following ethical standards: compliance with all applicable laws, prohibition of child labor and forced labor, maintenance of safe and healthy working conditions, non-discrimination in employment practices, and protection of intellectual property rights.

Anti-Corruption and Anti-Bribery
Vendors must not offer gifts, entertainment, or other items of value to Acme Corp employees with the intent to influence business decisions. Vendors found to be engaging in bribery or corruption will be immediately removed from the approved vendor registry.

Data Privacy and Information Security for Vendors
Vendors with access to Acme Corp data must comply with Acme Data Classification and Handling Policy. Cloud vendors must process Acme data only in approved regions. Vendors must notify Acme within 72 hours of any data breach affecting Acme Corp data.
""",
        "data_privacy_policy.txt": """\
Data Privacy and Protection Policy
Document: DPP-2026 | Owner: Legal and Compliance

Data Classification
Acme Corp classifies data into four tiers: Public, Internal, Confidential, and Restricted. Cloud storage and processing of Confidential and Restricted data requires explicit approval from the CISO. AWS S3 buckets storing Confidential data must have server-side encryption enabled, access logging active, and public access blocked.

GDPR Compliance
Acme Corp processes personal data of EU data subjects in compliance with GDPR. All cloud vendors processing EU personal data must have a valid Data Processing Agreement in place. AWS, GCP, Azure, and Cloudflare all have DPAs available.

Data Retention
Financial records including vendor invoices must be retained for a minimum of 7 years per IRS regulations. Cloud-stored financial data must have lifecycle policies configured to prevent accidental deletion.
""",
        "information_security_policy.txt": """\
Information Security Policy
Document: ISP-2026 | Owner: CISO

Access Control Requirements
All cloud infrastructure access must follow the principle of least privilege. IAM roles for AWS must be reviewed quarterly and unnecessary permissions revoked. All privileged access must use multi-factor authentication. Access credentials must never be stored in source code repositories.

Incident Response
Security incidents affecting cloud infrastructure must be reported to the Security Operations Center within 1 hour of detection. Critical incidents including data breach, ransomware, and unauthorized access to financial systems require immediate CISO notification.

Security Review for Cloud Spend
All new cloud services or significant architecture changes with greater than $5,000 monthly impact require a security review by the cloud security team before deployment.
""",
        "acceptable_use_policy.txt": """\
Acceptable Use Policy for Corporate Resources
Document: AUP-2026 | Owner: IT and HR

Cloud Resource Usage
Corporate cloud accounts including AWS, GCP, and Azure are provided for business use only. Employees may not use corporate cloud accounts to run personal projects, cryptocurrency mining, or any non-business workloads. Automated monitoring is in place to detect anomalous resource usage.

SaaS Tool Usage
Approved SaaS tools including Slack, GitHub, and Datadog must be used only for business purposes and only with corporate accounts. Personal accounts must not be connected to corporate data sources.

Expense Reporting
Corporate credit cards and expense reports are subject to audit. Submitting false or inaccurate expense reports is grounds for immediate termination. Employees must retain receipts for all expenses above $25. Expense reports must be submitted within 30 days of the expenditure date.
""",
        "contract_review_guidelines.txt": """\
Contract Review and Approval Guidelines
Document: CRG-2026 | Owner: Legal Department

Contract Value Thresholds
Contracts with annual value below $10,000 may be executed by department managers using pre-approved standard terms. Contracts between $10,000 and $100,000 require VP signature. Contracts above $100,000 require CFO signature. Cloud infrastructure contracts such as AWS Enterprise Discount Program and Cloudflare Enterprise are typically multi-year and fall under the CFO signature threshold.

Payment Terms Review
Legal must verify that all contracts specify Net-30 payment terms or receive CFO exception approval for non-standard terms. Contracts with Due Upon Receipt or Immediate Payment terms that were not approved by the CFO are not binding on Acme Corp and the AP team will not process associated invoices.

Renewal and Amendment
Contract amendments increasing value above the original approval threshold must go through the full approval process for the new total value.
""",
        "expense_reimbursement_policy.txt": """\
Employee Expense Reimbursement Policy
Document: ERP-2026 | Owner: HR and Finance

Eligible Expenses
Eligible reimbursable expenses include: business travel including flights, hotels, and ground transport, client meals and entertainment with proper documentation, conference registration fees, approved SaaS subscriptions for individual use, and office supplies under $100.

Documentation Requirements
All expense claims must be supported by original receipts or electronic equivalents. For meals and entertainment over $50, a business purpose description and list of attendees is required.

Reimbursement Timeline
Properly submitted expense reports with all required documentation are reimbursed within 10 business days via direct deposit. Expense reports must be submitted within 30 days of the expenditure date.
""",
        "it_asset_management_policy.txt": """\
IT Asset Management Policy
Document: ITAMP-2026 | Owner: IT Department

Hardware Asset Lifecycle
All hardware assets over $500 are tracked in the IT Asset Management System. Assets are depreciated over their useful life per the accounting policy. Disposal of hardware assets requires IT sign-off to ensure data sanitization before equipment leaves Acme Corp premises.

Software Asset Tracking
All software licenses including SaaS subscriptions must be registered in the IT Asset Management System within 5 business days of purchase. Cloud service agreements including AWS, Cloudflare, Datadog, and Zoom are tracked as software assets with monthly cost updates from the cloud cost management tool.

Cloud Resource Tagging
All cloud resources must be tagged with mandatory tags: Department such as ENG-01, Project, Environment, CostCenter, and Owner. Resources missing mandatory tags are automatically flagged and the resource owner manager is notified.
""",
        "change_management_policy.txt": """\
Change Management Policy
Document: CMP-2026 | Owner: IT Operations

Change Classification
Changes are classified as Standard which are pre-approved low-risk routine changes, Normal which requires CAB approval for any non-standard change, and Emergency which covers critical service restoration with expedited approval. Cloud infrastructure changes affecting production workloads are classified as Normal changes by default and require CAB approval with 5 business days notice.

Financial Impact Assessment
All Normal changes with an estimated cost impact above $5,000 must include a financial impact assessment signed off by the VP of Engineering and reviewed by Finance. This assessment must document monthly recurring cost, one-time setup costs, expected ROI, and alternatives considered.

Cloud Vendor Changes
Changes to cloud vendor agreements including new services, tier changes, and regional expansions must be communicated to Finance and Procurement at least 30 days in advance so that budget planning and contract reviews can be completed.
""",
        "business_continuity_policy.txt": """\
Business Continuity and Disaster Recovery Policy
Document: BCP-2026 | Owner: Operations and IT

Cloud Provider Redundancy Requirements
Mission-critical applications must be deployed across a minimum of two AWS availability zones. Applications with a Recovery Time Objective of less than 4 hours must have automated failover configured. Cloudflare is used as the primary DDoS protection and CDN layer.

Financial Systems Continuity
The AP system and ERP must maintain 99.9 percent availability during business hours Monday through Friday 8am to 6pm local time. Planned maintenance windows are scheduled on weekends with at least 5 business days notice.

Vendor Payment During Disruptions
In the event of a system disruption preventing normal AP processing, emergency manual payment authorization may be granted by the CFO for critical vendor payments. All emergency payments must be documented and reconciled in the ERP within 5 business days of service restoration.
""",
        "vendor_onboarding_checklist.txt": """\
Vendor Onboarding Requirements and Checklist
Document: VOC-2026 | Owner: Procurement

Required Documentation for New Vendors
Before any payment can be made to a new vendor, the following must be completed: W-9 or W-8BEN on file for tax reporting purposes, vendor registration in the Approved Vendor Registry, bank account verification, NDA signed if vendor will access confidential information, and Master Service Agreement or Purchase Order with payment terms specified as Net-30.

Cloud Vendor Specific Requirements
Cloud vendors must additionally provide: annual SOC 2 Type II report, data processing agreement for GDPR compliance, service level agreement with uptime guarantees and credit terms, and billing contact information for invoice disputes. All cloud vendor invoices must reference the Acme Corp enterprise account number.

Onboarding Timeline
Standard vendor onboarding takes 5 to 10 business days. Expedited onboarding available for urgent business needs with VP of Finance approval. New vendors who submit invoices before completing onboarding will have payments held until onboarding is complete.
""",
        "payment_dispute_resolution.txt": """\
Payment Dispute Resolution Policy
Document: PDR-2026 | Owner: Finance and Legal

Grounds for Disputing an Invoice
Acme Corp may dispute a vendor invoice under the following conditions: incorrect amount billed, services not rendered or goods not received, duplicate invoice for same period and services, non-standard payment terms not agreed in writing, invoice references an expired or invalid purchase order, or vendor not registered in the Approved Vendor Registry.

Dispute Process
The AP team will notify the vendor in writing within 5 business days of identifying a disputed invoice. The notification must specify the grounds for dispute and the corrective action required. The vendor has 15 business days to respond. If no response is received, the invoice is cancelled and returned.

Cloud Vendor Specific Disputes
Cloud vendor invoices such as those from AWS, Cloudflare, Google Cloud, and Azure may be disputed for: charges exceeding contracted rates, unexpected overage charges, services in unapproved regions, or payment terms that deviate from the master agreement. The Cloudflare invoice for $8,450 requires verification against the contracted annual enterprise rate before payment.
""",
    }

    doc_dir = Path("data/documents")
    for filename, content in docs.items():
        filepath = doc_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"  Generated: {len(docs)} policy documents in data/documents/ (plain .txt)")


if __name__ == "__main__":
    print("\n[*] Initializing Data Generator...\n")

    print("Creating directory structure:")
    create_directories()

    print("\nCleaning up legacy files:")
    cleanup_legacy_files()

    print("\nGenerating messy raw input files:")
    generate_messy_csv()
    generate_invoices()

    print("\nGenerating RAG policy documents (.txt):")
    generate_policy_documents()

    print("\n[OK] Data generation complete!")
    print("   -> 42 rows in messy_ledger.csv (7 different date formats, 5 departments)")
    print("   -> 8 messy invoice text files (OCR-style typos and formatting noise)")
    print("   -> 22 dense policy documents as plain .txt for FAISS indexing")
    print()
    print("Next steps:")
    print("   -> Run: pip install -r requirements.txt")
    print("   -> Run: python main.py")
