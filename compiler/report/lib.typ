#let report-template(
  title: "Report Title",
  subtitle: none,
  university: "Tecnológico de Monterrey",
  school-name: "Escuela de Ingeniería y Ciencias",
  authors: (),
  date: datetime.today(),
  school-color: rgb("#162773"),
  logo: image("tec-de-monterrey-4-logo-png-transparent.png"),
  abstract: none,
  body,
) = {
  set text(lang: "en")

  let overlay(img, color) = layout(bounds => {
    let size = measure(img, ..bounds)
    img
    place(top + left, block(..size, fill: color))
  })

  // --- Cover ---
  page(
    margin: 0cm,
    numbering: none,
  )[
    #if logo != none {
      place(
        right + bottom,
        dx: 50%,
        dy: 22%,
        block(overlay(
          logo,
          white.transparentize(15%),
        ), width: 190%),
      )
    }

    #place(
      left + top,
      rect(
        width: 2cm,
        height: 100%,
        fill: school-color,
      ),
    )

    #pad(
      left: 3cm,
      right: 2cm,
      top: 3cm,
      bottom: 3cm,
      {
        set text(size: 12pt)

        upper(university)
        linebreak()
        text(weight: "bold")[#upper(school-name)]
        linebreak()

        v(1fr)

        set text(size: 32pt, weight: "bold")
        set par(leading: 0.5em, justify: false)
        title
        set text(size: 20pt, weight: "regular")
        set par(leading: 0.5em, justify: false)
        subtitle

        v(1fr)
        set text(size: 14pt, weight: "regular")

        if authors.len() > 0 {
          [*Authors:* \ ]
          for author in authors {
            [#author \ ]
          }
        }

        linebreak()
        datetime.display(date)
      },
    )
  ]

  // --- Abstract ---
  if abstract != none {
    page(
      header: none,
      numbering: none,
    )[
      #set text(size: 12pt)

      #block(
        above: 0pt,
        below: 25pt,
        {
          set text(size: 30pt, weight: "regular")
          align(right)[Abstract]
          v(-0.25em)
          line(length: 100%, stroke: 0.5pt)
          v(0.5em)
        },
      )

      #set par(justify: true, leading: 0.65em)
      #abstract
    ]
  }

  // --- Table of contents ---
  page(
    header: none,
    numbering: none,
  )[
    #show outline.entry.where(level: 1): it => {
      v(18pt, weak: true)
      text(fill: school-color)[#strong(it)]
    }

    #show outline.entry.where(level: 2): it => {
      v(12pt, weak: true)
      text(fill: school-color)[#it]
    }

    #set text(size: 12pt)

    #block(
      above: 0pt,
      below: 25pt,
      {
        set text(size: 30pt, weight: "regular")
        align(right)[Contents]
        v(-0.25em)
        line(length: 100%, stroke: 0.5pt)
        v(0.5em)
      },
    )

    #set par(leading: 1.8em)

    #outline(
      title: none,
      depth: 2,
      indent: auto,
    )
  ]

  // --- Content ---
  body
}
