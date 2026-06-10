#import "lib.typ": report-template

#set document(title: "Report Template")
#set heading(numbering: "1.")
#set par(justify: true, spacing: 0.65em)

#show: report-template.with(
  title: "[Report title]",
  subtitle: "[Optional subtitle]",
  authors: (
    "[Author name]",
  ),
  abstract: [
    [Write a brief summary of the report: purpose, scope, methodology, key findings, and recommendations.]
  ],
)

#set table(
  fill: (x, y) => if y == 0 { rgb("#1a3c5e") } else { none },
  stroke: 0.8pt + rgb("#cccccc"),
)

#show table.cell.where(y: 0): set text(fill: white, weight: "bold")

= INTRODUCTION <introduction>
== Background <background>
[Describe the context and motivation for the report.]

== Objectives <objectives>
[State the general and specific objectives.]

== Scope <scope>
[Define the boundaries, assumptions, limitations, and intended audience.]

== Methodology <methodology>
[Summarize the methods, sources, tools, and criteria used to develop the report.]

= CONTEXT AND DESCRIPTION <context>
== Organization or system overview <overview>
[Describe the organization, process, product, system, or case being analyzed.]

== Relevant stakeholders <stakeholders>
[Identify relevant stakeholders, their needs, and expectations.]

#align(center)[
  #table(
    columns: 3,
    align: left,
    [*Stakeholder*], [*Need or expectation*], [*Planned response*],
    [[Stakeholder]], [[Need or expectation]], [[Response]],
  )
]

== Key constraints and assumptions <constraints>
[List relevant constraints, assumptions, dependencies, and exclusions.]

= PROCESS OR SYSTEM DESCRIPTION <process>
== General description <process-general>
[Describe the process, workflow, architecture, or system at a high level.]

== Flow diagram <flow-diagram>
[Insert a diagram, figure, or process map here if needed.]

== Inputs and outputs <inputs-outputs>
#align(center)[
  #table(
    columns: (1fr, 1.5fr, 1.5fr),
    inset: 10pt,
    align: horizon,
    [*Stage*], [*Inputs*], [*Outputs*],
    [[Stage]], [[Inputs]], [[Outputs]],
  )
]

== Critical points <critical-points>
[Identify the most important risks, bottlenecks, opportunities, or decision points.]

= ANALYSIS <analysis>
== Evaluation criteria <evaluation-criteria>
[Define the criteria, metrics, scoring system, or analytical framework.]

== Findings <findings>
[Present findings, observations, data, or evidence.]

#align(center)[
  #table(
    columns: 4,
    align: left,
    [*Item*], [*Criterion*], [*Result*], [*Notes*],
    [[Item]], [[Criterion]], [[Result]], [[Notes]],
  )
]

== Interpretation <interpretation>
[Explain what the results mean and why they matter.]

= REQUIREMENTS AND COMPLIANCE <requirements>
== Applicable requirements <applicable-requirements>
[List applicable requirements, standards, policies, regulations, or internal criteria.]

#align(center)[
  #table(
    columns: 4,
    align: left,
    [*Requirement*], [*Topic*], [*Applicability*], [*Evidence*],
    [[Requirement]], [[Topic]], [[Applicability]], [[Evidence]],
  )
]

== Risks and gaps <risks-gaps>
[Identify risks, gaps, nonconformities, or open questions.]

= PROPOSALS AND RECOMMENDATIONS <recommendations>
== Proposed actions <proposed-actions>
[Describe proposed actions, alternatives, or improvements.]

#align(center)[
  #table(
    columns: 4,
    align: left,
    [*Action*], [*Owner*], [*Priority*], [*Expected outcome*],
    [[Action]], [[Owner]], [[Priority]], [[Outcome]],
  )
]

== Cost-benefit or impact summary <impact-summary>
[Summarize expected costs, benefits, impact, feasibility, and trade-offs.]

== Implementation roadmap <roadmap>
#align(center)[
  #table(
    columns: (auto, 1fr, 1fr, auto),
    inset: 10pt,
    align: horizon,
    [*Phase*], [*Key activity*], [*Deliverable*], [*Timeline*],
    [[Phase 1]], [[Activity]], [[Deliverable]], [[Timeline]],
    [[Phase 2]], [[Activity]], [[Deliverable]], [[Timeline]],
    [[Phase 3]], [[Activity]], [[Deliverable]], [[Timeline]],
  )
]

= CONCLUSIONS <conclusions>
[Summarize the main conclusions, recommended next steps, and unresolved items.]
