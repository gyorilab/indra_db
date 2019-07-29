// CURATION FUNCTIONS

// Global constants
let ALL_COLLAPSED = true;

// Force activate the sub items of the table of contents after page load
$(document).ready(function () {
    $('a[href="#statements"]').addClass('active')
});

// Variables
let pubmed_fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";
let latestSubmission = {
    'ddSelect': '',
    'ev_hash': '',
    'source_hash': '',
    'submit_status': 0
};


// Turn on all the toggle buttons and connect them to a funciton.
document.addEventListener('DOMContentLoaded', () => {
   document.querySelectorAll('.curation_toggle')
       .forEach(function(toggle) {
           toggle.onclick = function() {
               addCurationRow(this.closest('tr'));
               this.onclick=null;
           };
           toggle.style.display = 'inline-block';
       })
});

// Check the API key input
function keySubmit(key_value) {
    let ensure_user = document.getElementById("ensure_user_on_api_key");
    // Default value still there or nothing entered
    if (key_value == "No key given." | !key_value) {
        ensure_user.textContent = "No key given.";
        // nothing entered
    } else {
        console.log("document.getElementById(\"api_key_input\").value: " + document.getElementById("api_key_input").value);
        ensure_user.textContent = "Key stored!";
    }
}

function submitButtonClick(clickEvent) {
    // Get the user's email and optionally api key.
    let jwt = localStorage.getItem('jwt_access');

    // If not in storage, reveal a popup
    if (!jwt) {
        const overlay = document.querySelector('#overlay');
        document.querySelector('#x-out').onclick = () => {
            // Abort the submit
            overlay.style.display = "none";
            return false;
        };
        document.querySelector('#overlay-form').onsubmit = function() {
            // Log the result
            console.log(`Got user email, ${this.email.value}, and password`);

            // Check for an email, if none, reject the form (do nothing)
            if (!this.email || !this.password) {
                // Ideally a warning or explanation should be given.
                console.log("Missing password or email.");
                return false;
            }

            let req = $.ajax({
                url: LOGIN_URL,
                type: 'POST',
                dataType: 'json',
                contentType: 'application/json',
                data: JSON.stringify({
                    'email': this.email.value,
                    'password': this.password.value
                })
            });

            if (req.status != 200) {
                // Ideally explain what went wrong.
                console.log(`Error authenticating: ${req.responseJSON}`)
                return false;
            }

            // Store the results
            localStorage.setItem('jwt_access', req.responseJSON.access_token);
            localStorage.setItem('jwt_refres', req.responseJSON.refresh_token);

            // Hide the overlay again.
            overlay.style.display = "none";

            // Call the function again (this time it will go past here).
            submitButtonClick(clickEvent);

            /// Make sure nothing else unwanted happens.
            return false;
        };
        overlay.style.display = "block";
        return false;
    }

    // Get mouseclick target, then parent's parent
    let pn = clickEvent.target.parentNode.parentNode;
    let btn_row_tag = pn.closest('tr');
    let s = pn.getElementsByClassName("dropdown")[0]
        .getElementsByTagName("select")[0];

    // DROPDOWN SELECTION
    let err_select = s.options[s.selectedIndex].value;
    if (!err_select) {
        alert('Please select an error type or "correct" for the statement in the dropdown menu');
        return;
    }

    // TEXT BOX CONTENT
    // Get "form" node, then the value of "input"
    let user_text = pn
        .getElementsByClassName("form")[0]
        .getElementsByTagName("input")[0]
        .value;

    // Refuse submission if 'other' is selected without providing a description
    if (!user_text && err_select === "other") {
        alert('Must describe error when using option "other..."!');
        return;
    }

    // GET REFERENCE TO STATUS BOX (EMPTY UNTIL STATUS RECEIVED)
    let statusBox = pn.getElementsByClassName("submission_status")[0].children[0];

    // Step back to the preceding tr tag
    let pmid_row = btn_row_tag.previousElementSibling;

    // PMID
    // Get pmid_linktext content
    let pmid_text = pmid_row
        .getElementsByClassName("pmid_link")[0]
        .textContent.trim();

    // Icon 
    let icon = pmid_row.getElementsByClassName("curation_toggle")[0];

    // HASHES: source_hash & stmt_hash
    // source_hash == ev['source_hash'] == pmid_row.id; "evidence level"
    const source_hash = pmid_row.id;
    // stmt_hash == hash == stmt_info['hash'] == table ID; "(pa-) statement level"
    const stmt_hash = pmid_row.parentElement.parentElement.id;

    // CURATION DICT
    // example: curation_dict = {'tag': 'Reading', 'text': '"3200 A" is picked up as an agent.', 'curator': 'Klas', 'ev_hash': ev_hash};
    let cur_dict = {
        'tag': err_select,
        'text': user_text,
        'curator': user_email,
        'ev_hash': source_hash
    };

    // console.log("source hash: " + source_hash)
    // console.log("stmt hash: " + stmt_hash)
    // console.log("Error selected: " + err_select);
    // console.log("User feedback: " + user_text);
    // console.log("PMID: " + pmid_text);
    // console.log("cur_dict");
    // console.log(cur_dict);

    // SPAM CONTROL: preventing multiple clicks of the same curation in a row
    // If the new submission matches latest submission AND the latest submission was
    // successfully submitted, ignore the new submission
    if (latestSubmission['ddSelect'] === err_select &
        latestSubmission['source_hash'] === source_hash &
        latestSubmission['stmt_hash'] === stmt_hash &
        latestSubmission['submit_status'] === 200) {
        alert('Already submitted curation successfully!');
        return;
    } else {
        latestSubmission['ddSelect'] = err_select;
        latestSubmission['source_hash'] = source_hash;
        latestSubmission['stmt_hash'] = stmt_hash;
    }
    let testing = false; // Set to true to test the curation endpoint of the API
    let ajx_response = submitCuration(cur_dict, stmt_hash, statusBox, icon, testing);
    console.log("ajax response from submission: ");
    console.log(ajx_response);
}
// Submit curation
function submitCuration(curation_dict, hash, statusBox, icon, test) {

    let _url = CURATION_ADDR + hash;

    if (test) {
        console.log("Submitting test curation...");
        _url += "&test";
    }
    // console.log("api key: " + api_key)
    console.log("url: " + _url);

    return $.ajax({
        url: _url,
        type: "POST",
        dataType: "json",
        header: {'Authorization': `Bearer ${localStorage.getItem('jwt_access')}`},
        contentType: "application/json",
        data: JSON.stringify(curation_dict),
        complete: function (xhr, statusText) {
            latestSubmission['submit_status'] = xhr.status;
            switch (xhr.status) {
                case 200:
                    statusBox.textContent = "Curation submitted successfully!";
                    icon.style = "color: #00FF00"; // Brightest green
                    break;
                case 400:
                    statusBox.textContent = xhr.status + ": Bad Curation Data";
                    icon.style = "color: #FF0000"; // Super red
                    break;
                case 404:
                    statusBox.textContent = xhr.status + ": Bad Link";
                    icon.style = "color: #FF0000";
                    break;
                case 500:
                    statusBox.textContent = xhr.status + ": Internal Server Error";
                    icon.style = "color: #FF0000";
                    break;
                case 504:
                    statusBox.textContent = xhr.status + ": Server Timeout";
                    icon.style = "color: #58D3F7"; // Icy blue
                    break;
                default:
                    console.log("Uncaught submission error: check ajax response");
                    console.log("xhr: ");
                    console.log(xhr);
                    statusBox.textContent = "Uncaught submission error; Code " + xhr.status;
                    icon.style = "color: #FF8000"; // Warning orange
                    break;
            }
        }
    });
}
// Creates the dropdown div with the following structure
// <div class="dropdown" 
//      style="display:inline-block; vertical-align: middle;">
//     <select>
//         <option value="" selected disabled hidden>
//             Select error type...
//         </option>
//         <option value="correct">Correct</option>
//         <option value="entity_boundaries">Entity Boundaries</option>
//         <option value="grounding">Grounding</option>
//         <option value="no_relation">No Relation</option>
//         <option value="wrong_relation">Wrong Relation</option>
//         <option value="act_vs_amt">Activity vs. Amount</option>
//         <option value="polarity">Polarity</option>
//         <option value="negative_result">Negative Result</option>
//         <option value="hypothesis">Hypothesis</option>
//         <option value="agent_conditions">Agent Conditions</option>
//         <option value="mod_site">Modification Site</option>
//         <option value="other">Other...</option>
//     </select>
// </div>
function createDDDiv() {
    let ddContainer = document.createElement("div");
    ddContainer.className = "dropdown";
    ddContainer.style = "display:inline-block; vertical-align: middle; margin-left: 9%;";

    let ddSelect = document.createElement("select");

    // DROPDOWN OPTIONS
    // Default; This is the option placeholder
    const placeholderOption = document.createElement("option");
    placeholderOption.value = "";
    placeholderOption.selected = "selected";
    placeholderOption.disabled = "disabled";
    placeholderOption.hidden = "hidden";
    placeholderOption.textContent = "Select error type...";
    ddSelect.appendChild(placeholderOption);
    // 1 "correct" No Error;
    option1 = document.createElement("option");
    option1.value = "correct";
    option1.textContent = "Correct";
    ddSelect.appendChild(option1);
    // 2 "entity_boundaries" Entity Boundaries;
    option2 = document.createElement("option");
    option2.value = "entity_boundaries";
    option2.textContent = "Entity Boundaries";
    ddSelect.appendChild(option2);
    // 3 "grounding" Grounding;
    option3 = document.createElement("option");
    option3.value = "grounding";
    option3.textContent = "Grounding";
    ddSelect.appendChild(option3);
    // 4 "no_relation" No Relation;
    option4 = document.createElement("option");
    option4.value = "no_relation";
    option4.textContent = "No Relation";
    ddSelect.appendChild(option4);
    // 5 "wrong_relation" Wrong Relation Type;
    option5 = document.createElement("option");
    option5.value = "wrong_relation";
    option5.textContent = "Wrong Relation";
    ddSelect.appendChild(option5);
    // 6 "act_vs_amt" Activity vs. Amount
    option6 = document.createElement("option");
    option6.value = "act_vs_amt";
    option6.textContent = "Activity vs. Amount";
    ddSelect.appendChild(option6);
    // 7 "polarity" Polarity;
    option7 = document.createElement("option");
    option7.value = "polarity";
    option7.textContent = "Polarity";
    ddSelect.appendChild(option7);
    // 8 "negative_result" Negative Result;
    option8 = document.createElement("option");
    option8.value = "negative_result";
    option8.textContent = "Negative Result";
    ddSelect.appendChild(option8);
    // 9 "hypothesis" Hypothesis;
    option9 = document.createElement("option");
    option9.value = "hypothesis";
    option9.textContent = "Hypothesis";
    ddSelect.appendChild(option9);
    // 10 "agent_conditions" Agent Conditions;
    option10 = document.createElement("option");
    option10.value = "agent_conditions";
    option10.textContent = "Agent Conditions";
    ddSelect.appendChild(option10);
    // 11 "mod_site" Modification Site;
    option11 = document.createElement("option");
    option11.value = "mod_site";
    option11.textContent = "Modification Site";
    ddSelect.appendChild(option11);
    // 12 "other" Other...
    option12 = document.createElement("option");
    option12.value = "other";
    option12.textContent = "Other...";
    ddSelect.appendChild(option12);
    // Add more options by following the structure above
    ddContainer.appendChild(ddSelect);
    return ddContainer;
}
// Creates the text box div with the following structure:
// <div class="form" 
//      style="display:inline-block; 
//             vertical-align: middle; 
//             top: 0px">
//     <form name="user_feedback_form">
//         <input type="text" 
//                maxlength="240"
//                name="user_feedback" 
//                placeholder="Optional description (240 chars)" 
//                value=""
//                style="width: 360px;">
//     </form>
// </div>
function createTBDiv() {
    let tbContainer = document.createElement("div");
    tbContainer.className = "form";
    tbContainer.style = "display:inline-block; vertical-align: middle; margin-left: 4%;";

    let tbForm = document.createElement("form");
    tbForm.name = "user_feedback_form";

    let tbInput = document.createElement("input");
    tbInput.type = "text";
    tbInput.maxlength = "240";
    tbInput.name = "user_feedback";
    tbInput.placeholder = "Optional description (240 chars)";
    tbInput.value = "";
    tbInput.style = "width: 360px;";

    tbForm.appendChild(tbInput);

    tbContainer.appendChild(tbForm);

    return tbContainer;
}
// Creates the submit button div with the following structure
// <div class="curation_button"
//      style="display:inline-block; 
//             vertical-align: middle;">
//     <button
//         type="button"
//         class="btn btn-default btn-submit pull-right"
//         style="padding: 2px 6px">Submit
//     </button>
//     < script tag type="text/javascript">
//     $(".btn-submit").off("click").on("click", function(b){
//         // Get parent node
//         parent_node = b.target.parentNode.parentNode
//         // Get reference to closest row tag (jquery)
//         this_row = $(this).closest("tr")
//         submitButtonClick(clickEvent)
//     })
//     </ script tag>
// </div>
function createSBDiv() {
    let sbContainer = document.createElement("div");
    sbContainer.className = "curation_button";
    sbContainer.style = "display:inline-block; vertical-align: middle; margin-left: 4%;";

    let sbButton = document.createElement("button");
    sbButton.type = "button";
    sbButton.className = "btn btn-default btn-submit pull-right";
    sbButton.style = "padding: 2px 6px; border: solid 1px #878787;";
    sbButton.textContent = "Submit";
    sbButton.onclick = submitButtonClick; // ATTACHES SCRIPT TO BUTTON

    sbContainer.appendChild(sbButton);

    return sbContainer;
}
// Creates the textbox that tells the user the status of the submission
// <div class="submission_status"
//      style="display:inline-block; 
//             vertical-align: middle;">
// <a class="submission_status"></a>
// </div>
function createStatusDiv() {
    let statusContainer = document.createElement("div");
    statusContainer.className = "submission_status";
    statusContainer.style = "display:inline-block; vertical-align: middle; margin-left: 4%;";

    let textContainer = document.createElement("i");
    textContainer.textContent = "";

    statusContainer.appendChild(textContainer);

    return statusContainer;

}

// Append row to the row that executed the click
// <tr class="cchild" style="border-top: 1px solid #FFFFFF;">
// <td colspan="4" style="padding: 0px; border-top: 1px solid #FFFFFF;">
function curationRowGenerator() {
    // Create new row element
    let newRow = document.createElement('tr');
    newRow.innerHTML = null;
    newRow.className = "cchild";
    newRow.style = "border-top: 1px solid #FFFFFF;";

    // Create new td element
    let newTD = document.createElement('td');
    newTD.style = "padding: 0px; border-top: 1px solid #FFFFFF; white-space: nowrap; text-align: left;";
    newTD.setAttribute("colspan", "4");

    // Add dropdown div
    let dropdownDiv = createDDDiv();
    newTD.appendChild(dropdownDiv);
    // Add textbox 
    let textBoxDiv = createTBDiv();
    newTD.appendChild(textBoxDiv);
    // Add submit button
    let buttonDiv = createSBDiv();
    newTD.appendChild(buttonDiv);
    // Add submission response textbox
    let statusDiv = createStatusDiv();
    newTD.appendChild(statusDiv);

    // Add td to table row
    newRow.appendChild(newTD);

    return newRow;
}
// Adds the curation row to current
function addCurationRow(clickedRow) {
    // Generate new row
    let curationRow = curationRowGenerator();

    // Append new row to provided row
    clickedRow.parentNode.insertBefore(curationRow, clickedRow.nextSibling);
}

// Expand/collapse row
$(function () {
    $("td[class='curation_toggle']").click(function (event) {
        event.stopPropagation();
        let $target = $(event.target);
        if (event.target.dataset.clicked == "true") {
            // Toggle (animation duration in msec)
            $target.closest("tr").next().find("div").slideToggle(200);
            // First click event
        } else {
            // Stay down (animation duration in msec)
            $target.closest("tr").next().find("div").slideDown(400);

            // Change color of icon to light gray
            event.target.style = "color:#A4A4A4;";

            // Set clicked to true
            event.target.dataset.clicked = "true"
        }
    });
});
