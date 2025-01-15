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

      // Basic D3 visualization
      const margin = { top: 20, right: 20, bottom: 30, left: 60 };

      const mainContainer = document.createElement("div");
      mainContainer.style.display = "flex";
      mainContainer.style.flexDirection = "row";
      mainContainer.style.alignItems = "center"; // Center align items vertically
      mainContainer.style.justifyContent = "flex-start"; // Align items to the start
      element.appendChild(mainContainer);

      const width = 1000;
      const height = 500;

      const svg = d3
        .select(mainContainer)
        .append("svg")
        .attr(
          "viewBox",
          `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`
        )
        .attr("preserveAspectRatio", "xMidYMid meet")
        .style("width", "100%")
        .style("height", "auto");

      // Create scales
      const x = d3
        .scaleLinear()
        .domain([0, data.length - 1])
        .range([0, width]);

      const y = d3.scaleLinear().domain([0, maxEntropy]).range([height, 0]);

      // Add X axis with character labels
      const xAxis = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${height})`)
        .call(
          d3
            .axisBottom(x)
            .ticks(data.length)
            .tickFormat((i) => data[Math.round(i)]?.char || "")
        );

      xAxis.selectAll("text").style("font-size", "0.9rem");

      // Add Y axis with both the axis line and label
      svg
        .append("g")
        .attr("transform", `translate(${margin.left},0)`) // Adjusted for left margin
        .call(d3.axisLeft(y));

      svg
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left + 75)
        .attr("x", 0 - height / 2)
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Entropy (bits)");

      // Add the blue line that tracks entropy values
      const line = d3
        .line()
        .x((d, i) => x(i))
        .y((d) => y(d.entropy));

      svg
        .append("path")
        .datum(data)
        .attr("fill", "none")
        .attr("stroke", "steelblue")
        .attr("stroke-width", 1.5)
        .attr("d", line)
        .attr("transform", `translate(${margin.left},0)`);

      // Add red horizontal threshold line
      const thresholdLine = svg
        .append("line")
        .attr("class", "threshold-line")
        .attr("x1", 0)
        .attr("x2", width)
        .attr("stroke", "#e01e37")
        .attr("stroke-width", 1)
        .style("stroke-dasharray", "5 5")
        .attr("transform", `translate(${margin.left},0)`);

      // Create slider container
      const sliderContainer = document.createElement("div");
      sliderContainer.style.height = "100%";
      sliderContainer.style.display = "flex";
      sliderContainer.style.flexDirection = "column";
      sliderContainer.style.justifyContent = "center";
      sliderContainer.style.position = "relative";
      mainContainer.appendChild(sliderContainer);

      // Create slider
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = maxEntropy;
      slider.step = 0.01;
      slider.value = 1.0;
      slider.style.width = "100%";
      slider.style.height = "20px";
      slider.style.transformOrigin = "center center";
      slider.style.transform = "rotate(-90deg)";
      sliderContainer.appendChild(slider);

      // Handle touch events to allow slider movement without scrolling
      let startY;
      let startValue;
      const sensitivity = 0.2; // Adjust this value to control sensitivity
      slider.addEventListener("touchstart", (e) => {
        startY = e.touches[0].clientY;
        startValue = parseFloat(slider.value);
        e.preventDefault(); // Prevent default to stop scrolling
      });

      slider.addEventListener(
        "touchmove",
        (e) => {
          const currentY = e.touches[0].clientY;
          const deltaY = startY - currentY; // Calculate vertical movement
          const valueChange =
            ((deltaY * (slider.max - slider.min)) / slider.clientHeight) * sensitivity; // Calculate value change based on movement
          slider.value = Math.min(
            Math.max(startValue + valueChange, slider.min),
            slider.max
          ); // Update slider value within bounds
          updateThresholdLine();
          updateVerticalLines();
          e.preventDefault(); // Prevent default to stop scrolling
        },
        { passive: false }
      );

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
              .style("stroke-dasharray", "3 3")
              .attr("transform", `translate(${margin.left},0)`);
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
