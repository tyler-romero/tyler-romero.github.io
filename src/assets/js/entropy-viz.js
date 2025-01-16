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
      const margin = { top: 30, right: 30, bottom: 30, left: 60 };

      const mainContainer = document.createElement("div");
      mainContainer.style.display = "flex";
      mainContainer.style.flexDirection = "row";
      mainContainer.style.alignItems = "center"; // Center align items vertically
      mainContainer.style.justifyContent = "flex-start"; // Align items to the start
      element.appendChild(mainContainer);

      const width = 800;
      const height = 400;

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
      xAxis.selectAll("text").style("font-family", "Consolas");

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

      // Add threshold value text
      const thresholdText = svg
        .append("text")
        .attr("class", "threshold-text")
        .attr("x", width + margin.left + 5) // Position to the right of the line
        .attr("y", 0) // Will be updated in updateThresholdLine
        .attr("dy", "0.35em") // Vertically center the text
        .style("font-size", "0.8rem")
        .style("font-family", "Consolas")
        .style("fill", "#e01e37");

      // Create slider container
      const sliderContainer = document.createElement("div");
      sliderContainer.style.flexDirection = "column";
      sliderContainer.style.justifyContent = "center";
      sliderContainer.style.width = "10px";
      mainContainer.appendChild(sliderContainer);

      // Create slider
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0;
      slider.max = maxEntropy;
      slider.step = 0.01;
      slider.value = 2.0;
      slider.style.width = "175px";
      slider.style.transform = "rotate(-90deg) translateY(-60px) translateX(10px)";
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
        thresholdText.attr("y", y(thresholdValue)).text(thresholdValue.toFixed(2));
      };

      // Function to update vertical lines
      const updateVerticalLines = () => {
        const thresholdValue = parseFloat(slider.value);

        // Remove existing vertical lines
        svg.selectAll(".vertical-line").remove();

        // Add new vertical lines
        for (let i = 0; i < data.length; i++) {
          if (i > 0 && data[i - 1].entropy > thresholdValue) {
            svg
              .append("line")
              .attr("class", "vertical-line")
              .attr("x1", x(i) - (x(1) - x(0)) / 2) // Shift x position by half a tick
              .attr("x2", x(i) - (x(1) - x(0)) / 2) // Shift x position by half a tick
              .attr("y1", y(0))
              .attr("y2", y(maxEntropy))
              .attr("stroke", "grey")
              .attr("stroke-width", 1)
              .style("stroke-dasharray", "3 3")
              .attr("transform", `translate(${margin.left},0)`);
          }
        }
      };

      // Add text visualization container after mainContainer
      const textContainer = document.createElement("div");
      textContainer.style.marginTop = "20px";
      textContainer.style.fontFamily = "monospace";
      textContainer.style.fontSize = "16px";
      textContainer.style.whiteSpace = "pre-wrap";
      textContainer.style.wordBreak = "break-all";
      element.appendChild(textContainer);

      // Function to update text visualization
      const updateTextVisualization = () => {
        const thresholdValue = parseFloat(slider.value);
        let visualizedText = "Patched text:\t";
        const colors = ["#C2EEC7", "#F7DAB0", "#F7B2B2", "#AEDDF3", "#CDC0EF"];
        let colorIndex = 0;
        let nextColorIndex = 0;

        data.forEach((d, index) => {
          visualizedText += `<span style="background-color: ${colors[colorIndex]}">${d.char}</span>`;
          if (d.entropy > thresholdValue) {
            nextColorIndex = (colorIndex + 1) % colors.length;
          }
          colorIndex = nextColorIndex;
        });

        textContainer.innerHTML = visualizedText;
      };

      // Initial update
      updateThresholdLine();
      updateVerticalLines();
      updateTextVisualization();

      // Update threshold line and vertical lines on slider change
      slider.addEventListener("input", () => {
        updateThresholdLine();
        updateVerticalLines();
        updateTextVisualization();
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
