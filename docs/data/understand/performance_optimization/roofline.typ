#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart

#set page(width: auto, height: auto, margin: .5cm)
#set text(size: 10pt)

#cetz.canvas({
  import cetz.draw: *

  // Define colors
  let dark_green = rgb("#006400")
  let dark_blue = rgb("#00008B")
  let dark_pink = rgb("#C71585")
  
  set-style(axes: (stroke: .5pt, tick: (stroke: .5pt)), legend: none)

  plot.plot(
    size: (12, 8),
    name: "roofline",
    x-min: 0.1, x-max: 10000, x-mode: "log",
    y-min: 0, y-max: 200,
    x-tick-step: none, y-tick-step: none,
    {
      // Memory bandwidth line - solid diagonal (dark green)
      plot.add(((0.0, 0.0), (18.025, 144.2)), style: (stroke: (paint: dark_green, thickness: 1.5pt)))
      
      // Memory bandwidth line - dashed continuation (dark green)
      plot.add(((18.025, 144.2), (140, 200)), style: (stroke: (paint: dark_green, thickness: 1.5pt, dash: "dashed")))
      
      // Arithmetic bandwidth line - dashed left segment (dark blue)
      plot.add(((0.0, 144.2), (18.025, 144.2)), style: (stroke: (paint: dark_blue, thickness: 1.5pt, dash: "dashed")))
      
      // Arithmetic bandwidth line - solid right segment (dark blue)
      plot.add(((18.025, 144.2), (10000, 144.2)), style: (stroke: (paint: dark_blue, thickness: 1.5pt)))
      
      // Ridge point marker
      plot.add(((18.025, 144.2),), mark: "square", mark-size: 0.15, style: (stroke: black, fill: black))
      
      // Anchors for labels and vertical lines
      plot.add-anchor("ridge", (18.025, 144.2))
      plot.add-anchor("mem-label", (10, 80))
      plot.add-anchor("arith-label", (8000, 144.2))
      plot.add-anchor("intensity1", (3, 92))
      plot.add-anchor("intensity2", (250, 144.2))
    },
    x-label: "Operational Intensity [FLOPs/byte]",
    y-label: "Performance [FLOPs/second]",
  )

  // Ridge point label
  content("roofline.ridge", [Ridge point], anchor: "north-west", padding: .2)
  
  // Memory bandwidth label with subtitle
  content("roofline.ridge", [
    #set text(fill: dark_green)
    Memory Bandwidth \
    #text(size: 8pt)[\[bytes/second\]]
  ], anchor: "south-east")
  
  // Arithmetic bandwidth label
  content("roofline.arith-label", [
    #set text(fill: dark_blue)
    Arithmetic Bandwidth
  ], anchor: "south-east", padding: .2)
  
  // Vertical line at Arithmetic Intensity 1 (x=1)
  line("roofline.intensity1", ((), "|-", (1, 0)), stroke: (paint: dark_pink, dash: "dashed"))
  content(("roofline.intensity1"), [
    #set text(fill: dark_pink)
    Arithmetic Intensity 1 \
    #text(size: 8pt)[#align(center)[Memory-bound]]
  ], anchor: "north-west", padding: .2)
  
  // Vertical line at Arithmetic Intensity 2 (x=1000)
  line("roofline.intensity2", ((), "|-", (1000, 0)), stroke: (paint: dark_pink, dash: "dashed"))
  content((rel: (4.65, 0.0), to: "roofline.intensity1"), [
    #set text(fill: dark_pink)
    Arithmetic Intensity 2 \
    #text(size: 8pt)[#align(center)[Compute-bound]]
  ], anchor: "north-west", padding: .2)
})
