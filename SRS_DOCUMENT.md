
# Software Requirements Specification (SRS)

**Project:** Model-Based Workflow Generation System (Refactor)
**Version:** 2.0.0
**Status:** Draft

## 1. System Overview

The objective is to refactor the existing minimal web interface into a robust, production-ready platform. The system facilitates the creation of complex task workflows that utilize optimal models rather than relying solely on generic LLMs. The refactor focuses on modularity, security, and an enhanced User Experience (UX) centered around a "Task" and "Project" architecture.

---

## 2. Functional Requirements

### 2.1. Authentication & Authorization Module

**2.1.1. Access Control**

* **Guard Clauses:** The application must enforce a global authentication guard. Any attempt to access protected routes (Home, Tasks, Projects) without a valid session token must result in an immediate redirection to the Login interface.

**2.1.2. User Interface (Login/Register)**

* **Split Layout Design:** The authentication page shall utilize a split-screen layout:
* **Visual Panel:** A dedicated area (e.g., left or right half) displaying the brand logo, branding assets, or dynamic marketing copy.
* **Interaction Panel:** A clean, centered form area for input fields.


* **Form Functionality:**
* Standard email/username and password fields with client-side validation (email format, password complexity).
* A "Remember Me" checkbox. When checked, the system shall issue a persistent authentication token (e.g., long-lived HTTP-only cookie) to maintain the session across browser restarts.



**2.1.3. Backend Authentication Logic**

* **Credential Handling:** Passwords must never be stored in plain text. Use industrial-standard hashing algorithms (e.g., Argon2id or Bcrypt) with per-user salts.
* **Session Management:** Implement secure session management (JWT or server-side sessions) with protection against Session Hijacking and Fixation.
* **DevOps Utility (Database Seeding):** A standalone script must be provided to bootstrap the database. This script shall idempotently create a superuser account:
* **Username:** `admin`
* **Password:** `admin`
* *Note: This script must be restricted to Development/Staging environments or strictly guarded in Production.*



### 2.2. Dashboard (Home Page)

Upon successful authentication, the user is directed to the Landing Dashboard.

* **Recent Activity Feed:** A widget displaying the most recently created or modified "Tasks" and "Projects."
* **Quick Actions:** Direct navigation links/buttons to "Create New Task" or "Create New Project."
* **Navigation:** A persistent global navigation bar or sidebar providing access to:
* Home
* Task Library
* Project Library
* User Profile/Settings/Logout



### 2.3. Task Management Module

#### 2.3.1. Create Task Wizard

The task creation process shall be refactored into a **Multi-Step Wizard** (Stepper UI) to reduce cognitive load. The UI must indicate progress (e.g., "Step 1 of 4") and allow backward navigation to edit previous steps before final submission. Step transitions use **Fade animations** for smooth visual feedback.

* **Stage 1: Metadata Configuration**
* **Fields:**
* `Task Name` (Text, Required, Unique constraint recommended).
* `Description` (Rich Text or Markdown, Required).
* `Evaluation Metrics` (Multi-select, Required). Users select one or more evaluation metrics displayed as **Tag components**.

* **Form Layout:** All form fields follow a consistent format:
* Label positioned **above** the input field
* Placeholder text within the input describing expected content
* Required indicator (\*) for mandatory fields

* **Validation:** Next step is disabled until all required fields are populated.


* **Stage 2: Pipeline Configuration**
* **Data Source:** The UI must fetch available types from the `TASK_TYPES_AND_PIPELINES` definitions.
* **Layout:** A **flat, non-collapsible list** displaying pipeline tags grouped by category headers.
* **Selection:** Pipeline tags are **optional** and displayed using the universal **Tag component**.
* **Interaction:** Click to select/deselect tags. Selected tags are visually distinguished with accent color styling.


* **Stage 3: Dataset Ingestion (Query Configuration)**
* **Structure:** Split into two distinct sections:

* **3A: Validation Dataset**
  * Purpose: Used for model evolution/training and performance validation.
  * **Label Requirement:** Ground truth labels are **Mandatory**.
  * **Recommendation Alert:** Display guidance that "at least 5 validation queries are recommended for optimal performance."
  * **Query Management:** Each query supports:
    * Multi-file upload (Images, Audio, Video, Text) via drag-and-drop or file selection
    * ZIP file upload for batch query import
    * Collapsible query cards with preview of attached files
    * Rich text label input

* **3B: Test Dataset**
  * Purpose: Used for inference/testing.
  * **Label Requirement:** Ground truth labels are **Optional** (workflow generates predictions).
  * **Query Management:** Same file upload capabilities as validation queries.

* **Visual Design:** Uses consistent Tag components to indicate query type (Validation/Test).


* **Stage 4: Review & Finalization**
* **Layout:** Two-column split design:
  * **Left Column:** Task Name and Evaluation Metrics
  * **Right Column:** Task Description (full text)

* **Pipeline Tags Section:** Titled "Pipeline Tags (N)" where N is the count of selected tags.
* **Dataset Statistics:** Justified StatBox components showing:
  * Total queries count
  * Validation queries count
  * Test queries count
  * Total files count

* **Action:** A "Create Task" button.
* **Success State:** Upon successful API response:
* Display a success toast/banner.
* Present two distinct actions:
1. **"View Task":** Redirects to the Task Details page (see 2.3.2).
2. **"Generate Workflow":** Redirects to the Create Project wizard with the current task pre-selected. *Requirement: Initially render this button as Disabled/Greyed-out for Phase 1 release.*







#### 2.3.2. Task Details View (Read/Edit)

* **Layout:** Mirrors the visual structure of the "Stage 4 Review" screen for consistency.
* **Metadata:** Display creation date and "Last Modified" timestamp.
* **Editing Capabilities:**
* The view is read-only by default.
* Provide three distinct "Edit" buttons corresponding to the data sections: (1) Metadata, (2) Dataset, (3) Pipeline Tags.
* **Interaction:** Clicking an Edit button opens a **Modal/Dialog** with a dimmed backdrop, allowing isolated updates to that specific section without navigating away.



#### 2.3.3. Task Library (All Tasks Page)

* **Layout:** Two-column layout.
* **Left Panel (Faceted Search & Filter):**
* **Sort:** Toggle between "Date Created" and "Date Modified" (Ascending/Descending).
* **Filter Categories:** Pipeline Types (NLP, Vision, etc.) and Pipeline Tags.
* **Filter Logic:** A control to define how multiple tags are handled:
* *Contains One (OR):* Match tasks having at least one of the selected tags.
* *Contains All (AND):* Match tasks having all selected tags.
* *Exact Match:* Match tasks having exactly the selected set of tags and no others.




* **Right Panel (Results):**
* **Global Search:** Text input for searching by Task Name or Description.
* **Grid/List View:** Cards displaying high-level info (Name, primary tags, metric count, created date).





---

## 3. Non-Functional Requirements (NFRs)

### 3.1. User Experience (UX/UI)

* **Feedback Loops:** All async actions (login, create task, upload) must show loading spinners or progress bars.
* **Error Handling:** Form validation errors should be displayed inline (near the specific field) rather than a generic alert at the top.
* **Responsiveness:** The layout must adapt to desktop and tablet viewports.
* **Drag & Drop:** File upload zones in Stage 3 should support drag-and-drop functionality.
* **Animations:** Step transitions in wizards use Fade animations for smooth visual feedback.

### 3.2. Design System

**3.2.1. Color Palette**

The application uses a cohesive color palette with the following design tokens:

| Token | Hex Value | Usage |
|-------|-----------|-------|
| Light | `#EAEFEF` | Backgrounds, surfaces, cards |
| Muted | `#BFC9D1` | Borders, disabled states, secondary text |
| Dark | `#25343F` | Primary text, headings, navigation |
| Accent (Orange) | `#FF9B51` | CTAs, highlights, selected states, links |
| Validation (Green) | `#22C55E` | Validation dataset indicators |
| Test (Blue) | `#3B82F6` | Test dataset indicators |

**3.2.2. Component Library**

* **Tag Component:** Universal tag styling used across metrics, pipeline tags, and dataset indicators.
  * Variants: `default`, `selected`, `category`, `validation`, `test`
  * All variants use a consistent border + lighter background pattern
  * Supports click handlers and removable state

* **FormField Component:** Standardized form field wrapper ensuring consistent layout:
  * Label positioned above input
  * Optional description text
  * Required field indicator (\*)
  * Error messages without inline red highlighting (uses top-level alerts instead)

**3.2.3. Navigation**

* **Logo:** Uses SVG logotype with color variant on hover state
* **Active Tab Indicator:** Orange underline with `::after` pseudo-element, touching the bottom of the nav bar
* **Active/Hover State:** Uses accent (orange) color for focused navigation items
* **Profile Menu:** Dropdown with `disableScrollLock` to prevent layout shifts

**3.2.4. Search Behavior**

* **Pipeline Tags Search:** Normalizes search by removing dashes (`-`) from both the query and tag names to enable matching tags containing dashes (e.g., "text-to-image" matches "texttoimage" search)

### 3.3. Security

* **CSRF Protection:** Implement Cross-Site Request Forgery tokens on all state-changing forms.
* **Sanitization:** All user inputs (especially Task Description and Labels) must be sanitized to prevent XSS (Cross-Site Scripting) attacks.
* **Rate Limiting:** Implement API rate limiting on the Login endpoint to prevent brute-force attacks.

### 3.4. Performance

* **Lazy Loading:** Task cards and project lists should utilize pagination or infinite scroll to handle large datasets efficiently.
* **Optimistic UI:** When editing task details, the UI should reflect changes immediately while the server processes the request in the background (with rollback on failure).

### 3.5. Data Integrity

* **Upload Validation:** Restrict file uploads by MIME type (e.g., prevent `.exe` uploads) and enforce a maximum file size limit (e.g., 50MB per file) to protect server storage.

---

## 4. Technical Implementation Notes

### 4.1. Admin Seed Script Specifications

The requested script (`create_admin.py` or similar) should perform the following logic:

1. Check if a user with username `admin` exists.
2. If yes, log "Admin exists" and exit.
3. If no, hash the string "admin" using the production hashing configuration.
4. Insert the record into the `users` table.
