#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#set page(width: auto, height: auto, margin: .5cm)
#set text(size: 10pt)

#cetz.canvas({
  import cetz.draw: *

  set-style(axes: (stroke: .5pt, tick: (stroke: .5pt)), legend: none)

  plot.plot(
    size: (12, 8),
    name: "roofline",
    x-min: 0.1, x-max: 10000, x-mode: "log",
    y-min: 0, y-max: 200,
    x-tick-step: none, y-tick-step: none,
    {
      plot.add(((0.1, 0), (18.025, 144.2), (10000, 144.2)), style: (stroke: (paint: blue)))
      plot.add-anchor("membound", (1, 64))
      plot.add-anchor("compbound", (1000, 144.2))
      plot.add-anchor("ridge", (18.025, 144.2))
    },
    x-label: "Operational Intensity [FLOPS/byte]",
    y-label: "Performance [TFLOPS]",
  )

  line("roofline.ridge", ((), "|-", (18.025, 0)), name: "ridge-horiz-line", stroke: (paint: red, dash: "densely-dashed"))

  line("roofline.membound", ((), "|-", (1, 5)), mark: (start: ">"), name: "membound-arrow")
  content("membound-arrow.end", [Memory bandwidth ceiling], anchor: "south", padding: .1)

  line("roofline.compbound", ((), "|-", (1, 2)), mark: (start: ">"), name: "compbound-arrow")
  content("compbound-arrow.end", [Compute ceiling], anchor: "north", padding: .1)
})