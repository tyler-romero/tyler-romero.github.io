console.log("Entropy viz script loaded");
console.log("D3 available:", typeof d3 !== "undefined");

function initEntropyViz() {
  const vizElements = document.querySelectorAll("[data-entropy-viz]");

  vizElements.forEach(async (element) => {
    // Check if element has already been initialized
    if (element.dataset.initialized) return;

    try {
      console.log("Entropy data:", data);

      // Basic D3 visualization
      const margin = { top: 20, right: 20, bottom: 30, left: 40 };
      const width = 600 - margin.left - margin.right;
      const height = 400 - margin.top - margin.bottom;

      const svg = d3
        .select(element)
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      // Create scales
      const x = d3
        .scaleLinear() // Changed from scaleBand to scaleLinear
        .domain([0, data.length - 1])
        .range([0, width]);

      const y = d3
        .scaleLinear()
        .domain([0, 1]) // Entropy values typically between 0 and 1
        .range([height, 0]);

      // Add X axis with character labels
      svg
        .append("g")
        .attr("transform", `translate(0,${height})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(data.length)
            .tickFormat((i) => data[Math.round(i)]?.char || "")
        );

      // Add Y axis with both the axis line and label
      svg.append("g").call(d3.axisLeft(y));

      svg
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x", 0 - height / 2)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Entropy");

      // Add the line
      const line = d3
        .line()
        .x((d, i) => x(i)) // Use index for x position
        .y((d) => y(d.entropy));

      svg
        .append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", line);

      // Mark as initialized
      element.dataset.initialized = "true";
    } catch (error) {
      console.error("Error initializing entropy visualization:", error);
    }
  });
}

// Run on DOMContentLoaded
document.addEventListener("DOMContentLoaded", initEntropyViz);

// Also run immediately in case DOM is already loaded
initEntropyViz();
