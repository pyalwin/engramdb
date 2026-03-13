"""
Synthetic Contract Generator with Multi-Hop QA Ground Truth.

Generates realistic legal contracts with controlled cross-references,
defined terms, and hierarchical structure. Each contract comes with
multi-hop questions whose ground-truth reasoning chains are known
exactly, enabling rigorous evaluation without external data.

Contract types:
  - NDA (Non-Disclosure Agreement)
  - MSA (Master Service Agreement)
  - License Agreement
  - Employment Agreement
  - SaaS Agreement

Each template has 4-8 sections with deliberate cross-references and
defined terms that create 2-3 hop reasoning chains.
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class SyntheticQuestion:
    """A multi-hop question with known ground truth."""
    id: str
    contract_id: str
    question: str
    answer: str
    reasoning_chain: list[str]
    hop_count: int
    question_type: str
    evidence_sections: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Contract templates with deliberate multi-hop structure
# ---------------------------------------------------------------------------

NDA_TEMPLATE = """MUTUAL NON-DISCLOSURE AGREEMENT

This Mutual Non-Disclosure Agreement ("Agreement") is entered into as of {date}
by and between {party_a} ("Disclosing Party") and {party_b} ("Receiving Party").

1. Definitions

1.1 "Confidential Information" means any and all non-public, proprietary, or
trade secret information disclosed by either party to the other, whether orally,
in writing, or by inspection. Confidential Information includes but is not
limited to: technical data, trade secrets, business plans, financial information,
customer lists, and product roadmaps.

1.2 "Permitted Purpose" means the evaluation of a potential business relationship
between the parties as described in Section 3.

1.3 "Representatives" means a party's officers, directors, employees, agents,
advisors, and consultants who have a need to know the Confidential Information
for the Permitted Purpose.

2. Obligations of Receiving Party

2.1 The Receiving Party shall hold all Confidential Information in strict
confidence and shall not disclose Confidential Information to any third party
without the prior written consent of the Disclosing Party, except as permitted
under Section 2.3.

2.2 The Receiving Party shall use Confidential Information solely for the
Permitted Purpose as defined in Section 1.2 and shall not use it for any
other purpose.

2.3 The Receiving Party may disclose Confidential Information to its
Representatives, provided that such Representatives are bound by obligations
of confidentiality no less restrictive than those set forth in this Section 2.

3. Permitted Purpose and Scope

3.1 The parties intend to explore {business_purpose}. All disclosures of
Confidential Information shall be made solely in furtherance of this purpose.

3.2 Nothing in this Agreement shall be construed to grant any rights in or
license to the Confidential Information, except the limited right to use
such information for the Permitted Purpose.

4. Exclusions from Confidential Information

4.1 Confidential Information shall not include information that: (a) is or
becomes publicly available through no fault of the Receiving Party; (b) was
known to the Receiving Party prior to disclosure; (c) is independently
developed by the Receiving Party without use of Confidential Information;
or (d) is disclosed pursuant to a court order, provided the Receiving Party
gives prompt notice per Section 6.2.

5. Term and Termination

5.1 This Agreement shall remain in effect for a period of {term_years} years
from the date first written above, unless earlier terminated.

5.2 Either party may terminate this Agreement upon {notice_days} days' prior
written notice to the other party. Upon termination, the obligations set
forth in Section 2 shall survive for a period of {survival_years} years.

5.3 Upon termination or expiration of this Agreement, the Receiving Party
shall promptly return or destroy all Confidential Information as described
in Section 7.

6. Remedies

6.1 The parties acknowledge that any breach of Section 2 may cause
irreparable harm to the Disclosing Party and that monetary damages may be
inadequate. Accordingly, the Disclosing Party shall be entitled to seek
injunctive relief in addition to any other remedies available at law or
in equity.

6.2 In the event of a compelled disclosure under Section 4.1(d), the
Receiving Party shall provide prompt written notice to the Disclosing Party
and cooperate in seeking a protective order.

7. Return of Materials

7.1 Upon written request by the Disclosing Party, or upon termination of
this Agreement as set forth in Section 5.3, the Receiving Party shall
promptly return or destroy all tangible materials containing Confidential
Information and provide written certification of such return or destruction.

8. General Provisions

8.1 Governing Law. This Agreement shall be governed by the laws of {jurisdiction}.

8.2 Entire Agreement. This Agreement constitutes the entire agreement between
the parties with respect to the subject matter hereof and supersedes all prior
negotiations and understandings.

8.3 Assignment. Neither party may assign this Agreement without the prior
written consent of the other party. Any attempted assignment in violation of
this Section 8.3 shall be void.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
first written above.
"""

MSA_TEMPLATE = """MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is made effective as of {date}
between {party_a} ("Client") and {party_b} ("Service Provider").

ARTICLE I - DEFINITIONS

1.1 "Services" means the professional services to be performed by Service
Provider as described in one or more Statements of Work executed pursuant
to Section 2.1.

1.2 "Deliverables" means all work product, reports, documentation, and
other materials created by Service Provider in connection with the Services,
as specified in the applicable Statement of Work.

1.3 "Intellectual Property" means all patents, copyrights, trademarks,
trade secrets, and other proprietary rights in and to the Deliverables,
subject to the ownership provisions in Article IV.

1.4 "Confidential Information" means any proprietary information disclosed
by either party, subject to the exclusions in Section 5.3.

ARTICLE II - SERVICES AND STATEMENTS OF WORK

2.1 Service Provider shall perform Services as set forth in mutually agreed
Statements of Work ("SOW"), each of which shall reference this Agreement and
be subject to its terms.

2.2 Each SOW shall include: (a) a description of Services; (b) the timeline
and milestones; (c) acceptance criteria for Deliverables as referenced in
Section 3.2; and (d) fees and payment terms per Article VI.

2.3 Service Provider shall perform all Services in a professional and
workmanlike manner, consistent with industry standards.

ARTICLE III - ACCEPTANCE AND DELIVERY

3.1 Service Provider shall deliver Deliverables in accordance with the
timeline specified in the applicable SOW per Section 2.2.

3.2 Client shall have {acceptance_days} business days to review each
Deliverable. If a Deliverable fails to meet the acceptance criteria defined
in the SOW, Client shall provide written notice specifying the deficiencies.

3.3 Service Provider shall correct any deficiencies within {correction_days}
business days of receiving notice under Section 3.2.

ARTICLE IV - INTELLECTUAL PROPERTY

4.1 All Deliverables and Intellectual Property created under this Agreement
shall be the exclusive property of Client, subject to the license granted
in Section 4.2.

4.2 Service Provider retains ownership of pre-existing tools, methodologies,
and frameworks ("Pre-Existing IP") used in performing the Services. Service
Provider grants Client a non-exclusive, perpetual license to use Pre-Existing
IP solely as embedded in the Deliverables.

4.3 Service Provider shall not incorporate any third-party intellectual
property into Deliverables without Client's prior written approval and
disclosure of applicable license terms per Section 5.1.

ARTICLE V - CONFIDENTIALITY

5.1 Each party agrees to hold Confidential Information of the other party
in strict confidence and to use such information only for the purposes of
this Agreement, as described in Article II.

5.2 The obligation of confidentiality shall survive termination of this
Agreement for a period of {confidentiality_years} years, consistent with
the survival provisions in Section 8.3.

5.3 Confidential Information excludes information that: (a) is publicly
available; (b) was known prior to disclosure; (c) is independently developed;
or (d) must be disclosed by law, provided notice is given per Section 8.5.

ARTICLE VI - FEES AND PAYMENT

6.1 Client shall pay Service Provider the fees specified in each SOW per
Section 2.2(d). Unless otherwise stated in the SOW, fees are due within
{payment_days} days of invoice.

6.2 Late payments shall accrue interest at the rate of {interest_rate}%
per month. If payment remains outstanding beyond {late_days} days,
Service Provider may suspend Services per Section 8.2.

ARTICLE VII - LIABILITY AND INDEMNIFICATION

7.1 Service Provider's total liability under this Agreement shall not exceed
the total fees paid by Client under the applicable SOW during the
{liability_months}-month period preceding the claim.

7.2 Service Provider shall indemnify Client against claims arising from:
(a) infringement of third-party Intellectual Property as described in
Article IV; (b) gross negligence or willful misconduct; or (c) breach of
confidentiality obligations under Article V.

7.3 IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR INDIRECT, INCIDENTAL,
SPECIAL, OR CONSEQUENTIAL DAMAGES.

ARTICLE VIII - TERM AND TERMINATION

8.1 This Agreement shall commence on the date first written above and shall
continue for an initial term of {term_years} years, with automatic renewal
for successive one-year periods unless terminated.

8.2 Either party may terminate this Agreement: (a) for convenience upon
{notice_days} days' written notice; (b) for cause if the other party
materially breaches and fails to cure within {cure_days} days of notice;
or (c) immediately upon the other party's insolvency.

8.3 Upon termination, the following shall survive: Article V (Confidentiality),
Article VII (Liability), this Section 8.3, and any accrued payment obligations
under Article VI.

8.4 Upon termination, Service Provider shall deliver all completed Deliverables
and work-in-progress as specified in Article III.

8.5 Any notices required under this Agreement shall be delivered in writing to
the addresses specified in Exhibit A.
"""

LICENSE_TEMPLATE = """SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of {date}
by and between {party_a} ("Licensor") and {party_b} ("Licensee").

1. Definitions

1.1 "Software" means the computer programs identified in Exhibit A, including
all updates and enhancements provided under this Agreement pursuant to
Section 4.

1.2 "Documentation" means the user manuals, technical specifications, and
other written materials provided with the Software as described in Section 2.3.

1.3 "Authorized Users" means Licensee's employees and contractors who are
authorized to use the Software, not to exceed {max_users} users as set forth
in Exhibit A.

1.4 "Fees" means the license fees and maintenance fees set forth in Exhibit B,
payable in accordance with Section 5.

2. License Grant

2.1 Subject to the terms of this Agreement and payment of Fees under Section 5,
Licensor grants Licensee a non-exclusive, non-transferable license to install
and use the Software solely for Licensee's internal business operations.

2.2 Licensee shall not: (a) sublicense, sell, or distribute the Software;
(b) modify or create derivative works; (c) reverse engineer or decompile the
Software; or (d) exceed the Authorized Users limit defined in Section 1.3.

2.3 Licensor shall provide Documentation sufficient for Authorized Users to
operate the Software. Documentation shall be updated concurrently with any
Software updates provided under Section 4.

3. Intellectual Property Rights

3.1 Licensor retains all right, title, and interest in the Software and
Documentation, including all Intellectual Property rights. Nothing in this
Agreement transfers ownership to Licensee, except as expressly licensed in
Section 2.1.

3.2 Licensee acknowledges that the Software contains trade secrets and
Confidential Information of Licensor, subject to the confidentiality
obligations in Section 6.

3.3 Any feedback, suggestions, or improvements provided by Licensee shall
become the property of Licensor, subject to Section 3.1.

4. Support and Maintenance

4.1 Licensor shall provide maintenance and support services ("Support") for
the Software during the term of this Agreement, subject to payment of the
maintenance fees specified in Exhibit B per Section 1.4.

4.2 Support includes: (a) bug fixes and patches; (b) minor version updates;
and (c) telephone and email support during business hours.

4.3 Major version upgrades are not included in Support and shall require a
separate agreement. Licensor shall provide reasonable notice of end-of-life
for any Software version per Section 7.4.

5. Fees and Payment

5.1 Licensee shall pay all Fees as set forth in Exhibit B. Annual license
fees are due within {payment_days} days of each anniversary of the Effective
Date.

5.2 If Licensee fails to pay any Fees within {grace_days} days of the due
date, Licensor may suspend access to the Software and Support per Section 7.3
until payment is received.

5.3 All Fees are non-refundable except as provided in Section 8.2.

6. Confidentiality

6.1 Licensee shall maintain the confidentiality of the Software source code
and any proprietary information disclosed under this Agreement, consistent
with the trade secret protections referenced in Section 3.2.

6.2 The confidentiality obligations in this Section 6 shall survive
termination of this Agreement for a period of {confidentiality_years} years.

7. Term and Termination

7.1 This Agreement is effective as of the date above and continues for an
initial term of {term_years} years.

7.2 Either party may terminate for material breach if the breach is not
cured within {cure_days} days of written notice. Material breach includes
failure to pay Fees per Section 5 and unauthorized use per Section 2.2.

7.3 Licensor may suspend the license immediately if Licensee: (a) fails to
pay Fees as described in Section 5.2; (b) exceeds the Authorized Users limit
per Section 1.3; or (c) violates the use restrictions in Section 2.2.

7.4 Upon termination, Licensee shall: (a) cease all use of the Software;
(b) return or destroy all copies of the Software and Documentation; and
(c) certify compliance in writing within {return_days} days.

8. Warranties and Limitations

8.1 Licensor warrants that the Software will conform to the Documentation
for a period of {warranty_months} months from delivery. Licensee's sole
remedy for breach of this warranty is repair or replacement per Section 4.

8.2 If Licensor cannot repair a material defect within {repair_days} days,
Licensee may terminate this Agreement and receive a pro-rata refund of
prepaid Fees under Section 5.3.

8.3 EXCEPT AS SET FORTH IN SECTION 8.1, THE SOFTWARE IS PROVIDED "AS IS."
LICENSOR DISCLAIMS ALL OTHER WARRANTIES.
"""

EMPLOYMENT_TEMPLATE = """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is made as of {date} between
{party_a} ("Company") and {party_b} ("Employee").

1. Position and Duties

1.1 Company hereby employs Employee as {job_title}. Employee shall report
to {supervisor_title} and perform the duties described in Exhibit A.

1.2 Employee shall devote full working time and best efforts to the
performance of duties, subject to the non-competition provisions in
Section 5.

2. Compensation and Benefits

2.1 Company shall pay Employee a base salary of ${salary:,} per year,
payable in accordance with Company's standard payroll practices.

2.2 Employee shall be eligible for an annual performance bonus of up to
{bonus_pct}% of base salary, subject to achievement of targets established
under Section 2.3.

2.3 Performance targets shall be established by {supervisor_title} within
30 days of each anniversary date and communicated in writing. Bonus
determination is in Company's sole discretion, subject to the terms of
Section 2.2.

3. Confidentiality and Intellectual Property

3.1 Employee acknowledges that during employment, Employee will have access
to Confidential Information as defined in the Company's Confidentiality
Policy attached as Exhibit B.

3.2 All inventions, works, and ideas ("Work Product") created by Employee
during employment and related to Company's business shall be the exclusive
property of Company.

3.3 Employee assigns to Company all rights in Work Product. The obligations
under this Section 3 shall survive termination per Section 6.4.

4. Term of Employment

4.1 Employment shall commence on {start_date} and continue until terminated
by either party pursuant to this Section 4.

4.2 Either party may terminate employment at any time for any reason upon
{notice_days} days' written notice to the other party.

4.3 Company may terminate employment immediately for Cause. "Cause" includes:
(a) material breach of this Agreement, including violations of Section 3 or
Section 5; (b) conviction of a felony; (c) willful misconduct; or
(d) failure to perform duties as described in Section 1.1 after written
warning.

5. Non-Competition and Non-Solicitation

5.1 During employment and for {non_compete_months} months following
termination, Employee shall not engage in any business that competes with
Company's business within {non_compete_radius} miles.

5.2 During employment and for {non_solicit_months} months following
termination, Employee shall not solicit Company's customers or employees.

5.3 If Employee is terminated without Cause under Section 4.2, Company shall
pay severance equal to {severance_months} months' base salary as
consideration for the obligations in this Section 5.

6. Termination Effects

6.1 Upon termination for any reason, Employee shall be entitled to: (a) earned
but unpaid base salary through the date of termination; (b) accrued but unused
vacation; and (c) any benefits required by law.

6.2 If terminated without Cause per Section 4.2, Employee shall receive
severance as described in Section 5.3, conditioned upon execution of a
release agreement.

6.3 Upon termination, Employee shall promptly return all Company property,
including all materials containing Confidential Information per Section 3.1.

6.4 The obligations in Section 3 (Confidentiality), Section 5 (Non-Competition),
and this Section 6 shall survive termination of this Agreement.
"""

SAAS_TEMPLATE = """SOFTWARE AS A SERVICE AGREEMENT

This Software as a Service Agreement ("Agreement") is made as of {date}
between {party_a} ("Provider") and {party_b} ("Customer").

1. Definitions

1.1 "Service" means the cloud-based software application described in
Exhibit A, accessible via the internet pursuant to Section 2.

1.2 "Customer Data" means all data, content, and information submitted by
Customer or its Authorized Users to the Service.

1.3 "Authorized Users" means Customer's employees and contractors authorized
to access the Service, not to exceed {max_users} users per the subscription
specified in Section 5.1.

1.4 "Service Level Agreement" or "SLA" means the uptime and performance
commitments set forth in Exhibit B, as referenced in Section 3.

2. Access and Use

2.1 Subject to payment of Fees under Section 5 and compliance with this
Agreement, Provider grants Customer a non-exclusive right to access and use
the Service during the Subscription Term.

2.2 Customer shall ensure that all Authorized Users comply with the
acceptable use policy described in Exhibit C. Customer is responsible for
its Authorized Users' compliance with Section 2.

2.3 Customer shall not: (a) sublicense or share access credentials;
(b) attempt to reverse engineer the Service; (c) use the Service for
unlawful purposes; or (d) exceed usage limits defined in Section 1.3.

3. Service Levels and Support

3.1 Provider shall maintain the Service in accordance with the SLA defined
in Exhibit B. In the event of failure to meet the SLA, Customer shall be
entitled to service credits as specified in Section 3.3.

3.2 Provider shall provide technical support during business hours.
Priority support is available for an additional fee per Exhibit B.

3.3 Service credits shall be calculated as follows: for each 1% below the
guaranteed uptime (as defined in the SLA per Section 1.4), Customer shall
receive a credit of {credit_pct}% of monthly Fees. Total credits shall not
exceed {max_credit_pct}% of monthly Fees.

4. Data and Security

4.1 Customer retains all rights in Customer Data. Provider shall use
Customer Data solely to provide the Service and as described in Section 2.

4.2 Provider shall implement security measures consistent with industry
standards to protect Customer Data, including encryption, access controls,
and regular audits as described in Exhibit D.

4.3 In the event of a data breach affecting Customer Data, Provider shall
notify Customer within {breach_notice_hours} hours and cooperate with
Customer's incident response, consistent with the obligations in Section 4.2.

4.4 Upon termination of this Agreement per Section 7, Provider shall make
Customer Data available for export for {export_days} days, after which
Provider may delete all Customer Data.

5. Fees and Payment

5.1 Customer shall pay the subscription Fees specified in Exhibit B.
Fees are based on the number of Authorized Users as defined in Section 1.3.

5.2 Fees are due {payment_frequency} in advance. Late payments shall accrue
interest at {interest_rate}% per month.

5.3 If Customer fails to pay within {grace_days} days, Provider may suspend
access to the Service per Section 7.3 until payment is received.

6. Intellectual Property

6.1 Provider retains all rights in the Service, including all software,
algorithms, and technology. Nothing in this Agreement transfers ownership
to Customer except as described in Section 4.1 regarding Customer Data.

6.2 Provider shall indemnify Customer against claims that the Service
infringes third-party intellectual property rights, subject to the
limitations in Section 8.

7. Term and Termination

7.1 The initial Subscription Term is {term_months} months beginning on
the Effective Date, with automatic renewal for successive {renewal_months}-month
periods unless either party provides {notice_days} days' notice.

7.2 Either party may terminate for material breach if not cured within
{cure_days} days of written notice. Material breach includes failure to
pay Fees per Section 5 and violations of Section 2.3.

7.3 Provider may suspend access immediately if Customer: (a) fails to
pay Fees per Section 5.3; (b) exceeds usage limits per Section 1.3; or
(c) violates acceptable use per Section 2.2.

7.4 Upon termination, the data export provisions in Section 4.4 shall apply.
The confidentiality obligations and limitation of liability shall survive
per Section 8.

8. Limitation of Liability

8.1 Provider's total liability shall not exceed the Fees paid by Customer
during the {liability_months}-month period preceding the claim.

8.2 NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL, SPECIAL, OR
CONSEQUENTIAL DAMAGES.

8.3 The limitations in this Section 8 shall not apply to: (a) breach of
confidentiality; (b) indemnification obligations under Section 6.2;
or (c) willful misconduct.
"""


# ---------------------------------------------------------------------------
# Template metadata: variables and multi-hop questions
# ---------------------------------------------------------------------------

TEMPLATES = {
    "nda": {
        "text": NDA_TEMPLATE,
        "variables": {
            "date": ["January 1, 2024", "March 15, 2024", "July 1, 2024", "October 10, 2024"],
            "party_a": ["Acme Corp", "TechVision Inc", "GlobalTrade LLC", "DataFlow Systems"],
            "party_b": ["Beta Industries", "Quantum Solutions", "NovaTech Partners", "Meridian Group"],
            "business_purpose": [
                "a potential strategic partnership",
                "a technology licensing arrangement",
                "a joint product development initiative",
            ],
            "term_years": ["2", "3", "5"],
            "notice_days": ["30", "60", "90"],
            "survival_years": ["3", "5", "7"],
            "jurisdiction": ["State of Delaware", "State of California", "State of New York"],
        },
        "questions": [
            {
                "question": 'If the Receiving Party shares Confidential Information with its employees, what conditions must be met under this agreement?',
                "answer": "Representatives may receive Confidential Information only if bound by confidentiality obligations no less restrictive than Section 2.",
                "reasoning_chain": ["1.3", "2.3", "2"],
                "hop_count": 2,
                "question_type": "definition_usage",
            },
            {
                "question": "What happens to confidentiality obligations after the agreement is terminated?",
                "answer": "Upon termination per Section 5.2, the obligations in Section 2 survive for the specified survival period.",
                "reasoning_chain": ["5.2", "2"],
                "hop_count": 2,
                "question_type": "cross_reference",
            },
            {
                "question": "If someone is compelled by court order to disclose Confidential Information, what must they do according to the agreement?",
                "answer": "Per Section 4.1(d), compelled disclosure is excluded from Confidential Information, but the Receiving Party must give prompt notice per Section 6.2 and cooperate in seeking a protective order.",
                "reasoning_chain": ["4.1", "6.2"],
                "hop_count": 2,
                "question_type": "conditional",
            },
            {
                "question": "What remedies are available if the Receiving Party breaches its confidentiality obligations, and what triggers those remedies?",
                "answer": "Breach of Section 2 obligations may cause irreparable harm, entitling the Disclosing Party to injunctive relief per Section 6.1.",
                "reasoning_chain": ["2", "6.1"],
                "hop_count": 2,
                "question_type": "cross_reference",
            },
            {
                "question": "What must happen with materials containing Confidential Information when the agreement ends?",
                "answer": "Upon termination per Section 5.3, the Receiving Party must return or destroy all materials per Section 7.1 and provide written certification.",
                "reasoning_chain": ["5.3", "7.1"],
                "hop_count": 2,
                "question_type": "termination_chain",
            },
            {
                "question": "Can the Receiving Party use Confidential Information for purposes other than the Permitted Purpose, and what defines that purpose?",
                "answer": "No. Section 2.2 restricts use to the Permitted Purpose, which is defined in Section 1.2 as evaluation of the business relationship described in Section 3.",
                "reasoning_chain": ["2.2", "1.2", "3"],
                "hop_count": 3,
                "question_type": "definition_usage",
            },
        ],
    },
    "msa": {
        "text": MSA_TEMPLATE,
        "variables": {
            "date": ["January 15, 2024", "April 1, 2024", "September 1, 2024"],
            "party_a": ["Enterprise Corp", "MegaTech Inc", "GlobalServices LLC"],
            "party_b": ["ConsultPro Solutions", "DevForce Inc", "AgileWorks Ltd"],
            "acceptance_days": ["10", "15", "20"],
            "correction_days": ["10", "15"],
            "confidentiality_years": ["3", "5"],
            "payment_days": ["30", "45", "60"],
            "interest_rate": ["1.5", "2.0"],
            "late_days": ["60", "90"],
            "liability_months": ["12", "24"],
            "term_years": ["2", "3"],
            "notice_days": ["30", "60"],
            "cure_days": ["30", "45"],
        },
        "questions": [
            {
                "question": "If a Deliverable fails acceptance testing, what is the complete process for resolution?",
                "answer": "Per Section 3.2, Client has a review period. If it fails, Client provides written notice of deficiencies. Service Provider must correct within the specified period per Section 3.3.",
                "reasoning_chain": ["2.2", "3.2", "3.3"],
                "hop_count": 3,
                "question_type": "cross_reference",
            },
            {
                "question": "What obligations survive termination of this agreement?",
                "answer": "Per Section 8.3, Article V (Confidentiality), Article VII (Liability), and accrued payment obligations under Article VI survive.",
                "reasoning_chain": ["8.3", "5", "7", "6"],
                "hop_count": 3,
                "question_type": "termination_chain",
            },
            {
                "question": "What protections does the Client have against third-party IP infringement in Deliverables?",
                "answer": "Section 4.3 requires prior approval for third-party IP. Section 7.2(a) provides indemnification for IP infringement related to Article IV.",
                "reasoning_chain": ["4.3", "7.2", "4"],
                "hop_count": 3,
                "question_type": "cross_reference",
            },
            {
                "question": "Under what conditions can Service Provider suspend services, and what payment obligations trigger this?",
                "answer": "Per Section 8.2, Service Provider may suspend for cause. Section 6.2 specifies that late payments beyond the grace period trigger suspension rights.",
                "reasoning_chain": ["6.2", "8.2"],
                "hop_count": 2,
                "question_type": "conditional",
            },
            {
                "question": "How are Confidential Information exclusions handled if disclosure is required by law?",
                "answer": "Section 5.3(d) excludes legally required disclosures, but notice must be given per Section 8.5.",
                "reasoning_chain": ["5.3", "8.5"],
                "hop_count": 2,
                "question_type": "cross_reference",
            },
        ],
    },
    "license": {
        "text": LICENSE_TEMPLATE,
        "variables": {
            "date": ["February 1, 2024", "June 15, 2024", "November 1, 2024"],
            "party_a": ["SoftCorp Inc", "CodeBase Technologies", "AppSuite LLC"],
            "party_b": ["UserTech Corp", "BizOps Industries", "DataDriven Inc"],
            "max_users": ["50", "100", "250", "500"],
            "payment_days": ["30", "45"],
            "grace_days": ["15", "30"],
            "confidentiality_years": ["5", "7"],
            "term_years": ["1", "3", "5"],
            "cure_days": ["30", "45", "60"],
            "return_days": ["15", "30"],
            "warranty_months": ["12", "24"],
            "repair_days": ["30", "60", "90"],
        },
        "questions": [
            {
                "question": "What happens if the Licensee fails to pay the required fees on time?",
                "answer": "Per Section 5.2, if fees are not paid within the grace period, Licensor may suspend access per Section 7.3.",
                "reasoning_chain": ["5.2", "7.3"],
                "hop_count": 2,
                "question_type": "conditional",
            },
            {
                "question": "What is the Licensee's remedy if the Software has a material defect that cannot be repaired?",
                "answer": "Per Section 8.1, the warranty provides for repair or replacement via Section 4. If repair fails within the specified period, Section 8.2 allows termination and a pro-rata refund under Section 5.3.",
                "reasoning_chain": ["8.1", "4", "8.2", "5.3"],
                "hop_count": 3,
                "question_type": "cross_reference",
            },
            {
                "question": "What confidentiality obligations apply to the Software, and how long do they last?",
                "answer": "Section 3.2 identifies the Software as containing trade secrets. Section 6.1 requires confidentiality consistent with Section 3.2. Per Section 6.2, these obligations survive termination.",
                "reasoning_chain": ["3.2", "6.1", "6.2"],
                "hop_count": 3,
                "question_type": "definition_usage",
            },
            {
                "question": "Under what circumstances can the Licensor immediately suspend the license?",
                "answer": "Per Section 7.3: (a) failure to pay per Section 5.2; (b) exceeding Authorized Users per Section 1.3; or (c) violating use restrictions per Section 2.2.",
                "reasoning_chain": ["7.3", "5.2", "1.3", "2.2"],
                "hop_count": 3,
                "question_type": "conditional",
            },
            {
                "question": "What must the Licensee do upon termination of this agreement?",
                "answer": "Per Section 7.4, Licensee must cease use, return or destroy all Software and Documentation copies, and certify compliance in writing.",
                "reasoning_chain": ["7.4", "7.2"],
                "hop_count": 2,
                "question_type": "termination_chain",
            },
        ],
    },
    "employment": {
        "text": EMPLOYMENT_TEMPLATE,
        "variables": {
            "date": ["January 2, 2024", "March 1, 2024", "August 15, 2024"],
            "party_a": ["TechCorp Inc", "InnovateCo", "FutureWorks LLC"],
            "party_b": ["Jane Smith", "John Doe", "Alex Johnson"],
            "job_title": ["Senior Software Engineer", "VP of Engineering", "Product Manager"],
            "supervisor_title": ["Chief Technology Officer", "VP of Engineering", "Director of Product"],
            "salary": [150000, 200000, 180000, 250000],
            "bonus_pct": ["15", "20", "25"],
            "start_date": ["February 1, 2024", "April 1, 2024", "September 1, 2024"],
            "notice_days": ["14", "30", "60"],
            "non_compete_months": ["12", "18", "24"],
            "non_compete_radius": ["25", "50"],
            "non_solicit_months": ["12", "18"],
            "severance_months": ["3", "6", "12"],
        },
        "questions": [
            {
                "question": "If the Employee is terminated without Cause, what compensation and obligations apply?",
                "answer": "Per Section 4.2, termination requires notice. Section 6.2 provides severance per Section 5.3, conditioned on a release. Non-compete and non-solicit per Section 5 still apply.",
                "reasoning_chain": ["4.2", "6.2", "5.3", "5"],
                "hop_count": 3,
                "question_type": "termination_chain",
            },
            {
                "question": "What constitutes Cause for termination, and what employment obligations does it relate to?",
                "answer": "Section 4.3 defines Cause including breach of Section 3 (Confidentiality) or Section 5 (Non-Competition), conviction of felony, willful misconduct, or failure to perform duties per Section 1.1.",
                "reasoning_chain": ["4.3", "3", "5", "1.1"],
                "hop_count": 3,
                "question_type": "definition_usage",
            },
            {
                "question": "What happens to intellectual property created by the Employee after termination?",
                "answer": "Per Section 3.2, all Work Product belongs to Company. Section 3.3 assigns all rights, and per Section 6.4, these obligations survive termination.",
                "reasoning_chain": ["3.2", "3.3", "6.4"],
                "hop_count": 3,
                "question_type": "cross_reference",
            },
            {
                "question": "What must the Employee return upon leaving the company?",
                "answer": "Per Section 6.3, Employee must return all Company property including materials containing Confidential Information per Section 3.1.",
                "reasoning_chain": ["6.3", "3.1"],
                "hop_count": 2,
                "question_type": "termination_chain",
            },
        ],
    },
    "saas": {
        "text": SAAS_TEMPLATE,
        "variables": {
            "date": ["January 10, 2024", "May 1, 2024", "October 1, 2024"],
            "party_a": ["CloudFirst Inc", "SaaSPro Technologies", "PlatformOne LLC"],
            "party_b": ["BusinessTech Corp", "DataOps Inc", "ScaleUp Industries"],
            "max_users": ["25", "100", "500", "1000"],
            "credit_pct": ["5", "10"],
            "max_credit_pct": ["25", "50"],
            "breach_notice_hours": ["24", "48", "72"],
            "export_days": ["30", "60", "90"],
            "payment_frequency": ["monthly", "quarterly", "annually"],
            "interest_rate": ["1.5", "2.0"],
            "grace_days": ["15", "30"],
            "term_months": ["12", "24", "36"],
            "renewal_months": ["12", "24"],
            "notice_days": ["30", "60", "90"],
            "cure_days": ["30", "45"],
            "liability_months": ["12", "24"],
        },
        "questions": [
            {
                "question": "What are the Customer's remedies if the Provider fails to meet the SLA?",
                "answer": "Per Section 3.1, Provider must meet SLA in Exhibit B. Section 3.3 provides service credits calculated per the formula, subject to the cap.",
                "reasoning_chain": ["3.1", "1.4", "3.3"],
                "hop_count": 3,
                "question_type": "cross_reference",
            },
            {
                "question": "What happens to Customer Data after the agreement is terminated?",
                "answer": "Per Section 7.4, the data export provisions in Section 4.4 apply: data is available for export for the specified period, after which Provider may delete it.",
                "reasoning_chain": ["7.4", "4.4"],
                "hop_count": 2,
                "question_type": "termination_chain",
            },
            {
                "question": "Under what conditions can Provider suspend Customer's access, and what triggers each condition?",
                "answer": "Section 7.3 allows suspension if: (a) payment failure per Section 5.3; (b) exceeding users per Section 1.3; or (c) acceptable use violations per Section 2.2.",
                "reasoning_chain": ["7.3", "5.3", "1.3", "2.2"],
                "hop_count": 3,
                "question_type": "conditional",
            },
            {
                "question": "What exceptions exist to the limitation of liability, and what provisions do they reference?",
                "answer": "Section 8.3 carves out: (a) breach of confidentiality; (b) indemnification under Section 6.2; and (c) willful misconduct.",
                "reasoning_chain": ["8.3", "6.2"],
                "hop_count": 2,
                "question_type": "cross_reference",
            },
            {
                "question": "How is the Customer responsible for its users' behavior on the platform?",
                "answer": "Section 2.2 makes Customer responsible for Authorized Users' compliance. Users must follow the acceptable use policy (Exhibit C). Section 1.3 defines and limits Authorized Users.",
                "reasoning_chain": ["2.2", "1.3", "2.3"],
                "hop_count": 3,
                "question_type": "definition_usage",
            },
        ],
    },
}


def generate_synthetic_dataset(
    num_variants_per_template: int = 3,
    seed: int = 42,
) -> dict:
    """
    Generate a complete synthetic multi-hop QA dataset.

    Args:
        num_variants_per_template: How many variable-substituted versions per template
        seed: Random seed for reproducibility

    Returns:
        Dataset dict with 'contracts' and 'questions' keys
    """
    rng = random.Random(seed)
    contracts = []
    questions = []

    for template_name, template_info in TEMPLATES.items():
        text_template = template_info["text"]
        variables = template_info["variables"]
        base_questions = template_info["questions"]

        for variant_idx in range(num_variants_per_template):
            # Pick random variable values
            values = {}
            for var_name, options in variables.items():
                values[var_name] = rng.choice(options)

            # Generate contract text
            try:
                contract_text = text_template.format(**values)
            except (KeyError, ValueError):
                contract_text = text_template
                for k, v in values.items():
                    contract_text = contract_text.replace("{" + k + "}", str(v))

            contract_id = f"{template_name}_{variant_idx:03d}"

            contracts.append({
                "id": contract_id,
                "text": contract_text,
                "type": template_name,
                "variables": {k: str(v) for k, v in values.items()},
            })

            # Generate questions for this contract
            for q_idx, q_template in enumerate(base_questions):
                questions.append({
                    "id": f"{contract_id}_q{q_idx:02d}",
                    "contract_id": contract_id,
                    "question": q_template["question"],
                    "answer": q_template["answer"],
                    "reasoning_chain": q_template["reasoning_chain"],
                    "hop_count": q_template["hop_count"],
                    "question_type": q_template["question_type"],
                    "evidence_sections": [],
                })

    return {
        "contracts": contracts,
        "questions": questions,
        "metadata": {
            "source": "EngramDB Synthetic Contract Generator",
            "total_contracts": len(contracts),
            "total_questions": len(questions),
            "templates": list(TEMPLATES.keys()),
            "num_variants_per_template": num_variants_per_template,
            "seed": seed,
        },
    }


def main():
    """Generate and save synthetic dataset."""
    print("=" * 60)
    print("Synthetic Contract Dataset Generator")
    print("=" * 60)

    dataset = generate_synthetic_dataset(num_variants_per_template=3, seed=42)

    print(f"\nGenerated {dataset['metadata']['total_contracts']} contracts")
    print(f"Generated {dataset['metadata']['total_questions']} questions")
    print(f"Templates: {', '.join(dataset['metadata']['templates'])}")

    # Stats by type
    type_counts: dict[str, int] = {}
    hop_counts: dict[int, int] = {}
    for q in dataset["questions"]:
        type_counts[q["question_type"]] = type_counts.get(q["question_type"], 0) + 1
        hop_counts[q["hop_count"]] = hop_counts.get(q["hop_count"], 0) + 1

    print("\n--- Question Types ---")
    for qtype, count in sorted(type_counts.items()):
        print(f"  {qtype}: {count}")

    print("\n--- Hop Counts ---")
    for hops, count in sorted(hop_counts.items()):
        print(f"  {hops}-hop: {count}")

    # Save
    output_dir = Path("data/cuad")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "multihop_qa_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Sample
    print("\n--- Sample Questions ---")
    for q in dataset["questions"][:3]:
        print(f"\n  [{q['question_type']}] ({q['hop_count']}-hop)")
        print(f"  Q: {q['question']}")
        print(f"  Chain: {' -> '.join(q['reasoning_chain'])}")


if __name__ == "__main__":
    main()
