# Solution Requirements

## Functional Requirements
- **User Authentication:** Users must be able to sign up, log in, and log out.
- **Loan Prediction:** System must accept applicant data and return an eligibility prediction.
- **Data Storage:** All applications and results must be stored in a NoSQL database (MongoDB).
- **History Tracking:** Users must be able to view their past applications.
- **Admin Analytics:** Admins need a dashboard to view overall application statistics.

## Non-Functional Requirements
- **Performance:** Prediction response time should be under 2 seconds.
- **Usability:** The UI must be responsive, modern, and intuitive.
- **Scalability:** The architecture should support future scaling (e.g., cloud deployment).
- **Reliability:** The ML model must have an accuracy of at least 80%.