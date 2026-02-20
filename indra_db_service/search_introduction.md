# INDRA DB Search Interface

On the landing page of the INDRA DB web interface, you can search for
interactions between entities extracted and assembled by INDRA represented as agents 
in [INDRA Statements](https://indra.readthedocs.io/en/latest/modules/Statements.html). 
The search interface allows you to filter the results by the interactions' INDRA 
Statement type, the MeSH terms associated with the publication from which the 
interactions were extracted, or by the publication from which the interactions were 
extracted. Once you have entered your search criteria, click the "Search" button to 
retrieve the Statements that match your search criteria.

## Search Options

The search options are as follows:

- Agent/Entity: enter one or, optionally, two agents. For each agent you can enter 
  either its identifier directly or enter its name and use the "Find identifier" button to 
  do a search for an identifier by the entity name. You can enter any type of agent 
  that INDRA recognizes, such as a gene, a small molecule, a biological process, etc.
- Agent role:
  - subject/object: the agent(s) can be both upstream/the controller and downstream/be 
    controlled in the retrieved Statements.
  - Subject: 
    - If one agent is entered: the agent is upstream/the controller in the retrieved 
      Statements.
    - If two agents are entered: the first agent (blue dot) is upstream/the controller 
      and the second agent (orange dot) is downstream/controlled in the retrieved 
      Statements.
  - Object: The roles are reversed compared to the Subject option:
    - If one agent is entered, the agent is downstream/controlled in the retrieved 
      Statements.
    - If two agents are entered, the first agent (blue dot) is downstream/controlled 
      and the second agent (orange dot) is upstream/the controller in the retrieved 
      Statements.
- Relation type: the type of Statement e.g. Activation, Phosphorylation, DecreaseAmount, 
  Complex, etc. Read more about the types of Statements in the 
  [INDRA documentation](https://indra.readthedocs.io/en/latest/modules/Statements.html).
  - Include subtypes checkbox: if checked, the search will include Statements of the specified relation type
    and all of its subtypes. For example, if you select "RegulateActivity" as the 
    relation type and check the "Include subtypes" box, the search will include 
    Statements of type "Activation" and "Inhibition".
- Context filter (MeSH): Enter the name or identifier of a **Me**dical **S**ubject 
  **H**eadings (MeSH) term that the papers retrieved as evidence are
  annotated with. If you only know the name of the MeSH term, you can also use the 
  grounding button to search for the MeSH term identifier.
- More filters - Filter by paper: Limit the search to a specific publication that 
  evidence comes from. To include multiple papers, select another paper search option 
  from the dropdown. In the paper search option, you can search by these publication 
  identifiers:
  - PMID: PubMed ID
  - PMCID: PubMed Central ID
  - DOI: Digital Object Identifier
  - TRID: Internal INDRA DB ID signifying a specific publication regardless of the
    external identifier (PMID, PMCID, DOI).
  - TCID: Internal INDRA DB ID signifying a piece of a text retrieved from 
    a particular source.

## Search Results

The search results are displayed in hierarchical list format. At the top level, the
most generic form of interaction matching the search criteria are displayed. Clicking
on one of the rows expands the next level of detail, showing the specific forms of
interactions that match the search criteria. Clicking on one of these rows expands the
next level of detail, showing the specific Statements that match the search criteria.
The nesting is at most three levels deep, but can also be less if e.g., there is only one
Statement type for one interaction type.

![Web UI screenshot](../doc/web_ui_results_expanded.png)
<span class="caption">Search results view with three levels of nesting expanded all 
the way down to the Statement evidence level for DUSP1 affecting MAPK1. The green dot 
over the pencil icon marks prior curations (see below) have been done for that piece of 
evidence</span>

The search results allows you to curate evidence for each Statement (requires login). 
To do this, click on the pencil icon next to the piece of evidence you want to curate. 
This will open a curation area where different options for curating the evidence are 
available. To read more about curation, see the
[curation tutorial](https://indra.readthedocs.io/en/latest/tutorials/html_curation.html)
in the INDRA documentation.
