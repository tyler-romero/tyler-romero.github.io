console.log("Entropy viz script loaded");
console.log("D3 available:", typeof d3 !== "undefined");

function initEntropyViz() {
  const vizElements = document.querySelectorAll("[data-entropy-viz]");

  vizElements.forEach(async (element) => {
    if (element.dataset.initialized) return; // end early if already initialized

    try {
      const response = await fetch("/assets/datasets/entropy-values.json");
      const data = await response.json();
      console.log("Fetched data:", data);

      // Find the maximum entropy value
      const maxEntropy = Math.max(...data.map((d) => d.entropy)) + 0.1;

      // Create slider container
      const sliderContainer = document.createElement("div");
      sliderContainer.style.marginBottom = "10px";
      element.insertBefore(sliderContainer, element.firstChild);

      // Create slider
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = maxEntropy; // Adjust max to match entropy range
      slider.step = 0.01; // Add steps for finer control
      slider.value = 1.0; // Set an initial value within the range
      sliderContainer.appendChild(slider);
      // access slider value with slider.value

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

      const y = d3.scaleLinear().domain([0, maxEntropy]).range([height, 0]);

      // Add X axis with character labels
      const xAxis = svg
        .append("g")
        .attr("transform", `translate(0,${height})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(data.length)
            .tickFormat((i) => data[Math.round(i)]?.char || "")
        );

      xAxis.selectAll("text").style("font-size", "0.8rem");

      // Add Y axis with both the axis line and label
      svg.append("g").call(d3.axisLeft(y));

      svg
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left)
        .attr("x", 0 - height / 2)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Entropy (bits)");

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

      // Add horizontal threshold line
      const thresholdLine = svg
        .append("line")
        .attr("class", "threshold-line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("stroke", "#e01e37")
        .attr("stroke-width", 1)
        .style("stroke-dasharray", "5 5");

      // Function to update threshold line position
      const updateThresholdLine = () => {
        const thresholdValue = parseFloat(slider.value);
        thresholdLine.attr("y1", y(thresholdValue)).attr("y2", y(thresholdValue));
      };

      // Function to update vertical lines
      const updateVerticalLines = () => {
        const thresholdValue = parseFloat(slider.value);

        // Remove existing vertical lines
        svg.selectAll(".vertical-line").remove();

        // Add new vertical lines
        data.forEach((d, originalIndex) => {
          if (d.entropy > thresholdValue) {
            svg
              .append("line")
              .attr("class", "vertical-line")
              .attr("x1", x(originalIndex))
              .attr("x2", x(originalIndex))
              .attr("y1", y(0))
              .attr("y2", y(maxEntropy))
              .attr("stroke", "grey")
              .attr("stroke-width", 1)
              .style("stroke-dasharray", "3 3");
          }
        });
      };

      // Initial update
      updateThresholdLine();
      updateVerticalLines();

      // Update threshold line and vertical lines on slider change
      slider.addEventListener("input", () => {
        updateThresholdLine();
        updateVerticalLines();
      });

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
