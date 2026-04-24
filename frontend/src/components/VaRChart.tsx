import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface VaRChartProps {
  returns: number[];
  var95: number;
  var99: number;
  cvar95: number;
  width?: number;
  height?: number;
}

const VaRChart: React.FC<VaRChartProps> = ({
  returns,
  var95,
  var99,
  cvar95,
  width = 600,
  height = 300,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || returns.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 30, right: 30, bottom: 50, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create histogram bins
    const minReturn = d3.min(returns) as number;
    const maxReturn = d3.max(returns) as number;
    const binWidth = (maxReturn - minReturn) / 50;

    const histogram = d3.bin<number, number>()
      .domain([minReturn, maxReturn])
      .thresholds(50);

    const bins = histogram(returns);

    const xScale = d3.scaleLinear()
      .domain([minReturn, maxReturn])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(bins, d => d.length) as number])
      .range([innerHeight, 0]);

    // Draw bars
    g.selectAll('rect.bar')
      .data(bins)
      .join('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.x0 as number))
      .attr('y', d => yScale(d.length))
      .attr('width', d => Math.max(0, xScale(d.x1 as number) - xScale(d.x0 as number) - 1))
      .attr('height', d => innerHeight - yScale(d.length))
      .attr('fill', d => {
        const midpoint = ((d.x0 as number) + (d.x1 as number)) / 2;
        if (midpoint <= -var99) return '#dc2626';
        if (midpoint <= -var95) return '#f97316';
        if (midpoint < 0) return '#64748b';
        return '#22c55e';
      })
      .attr('opacity', 0.8);

    // VaR lines
    const drawVarLine = (value: number, label: string, color: string, yOffset: number) => {
      g.append('line')
        .attr('x1', xScale(-value))
        .attr('x2', xScale(-value))
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', color)
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

      g.append('text')
        .attr('x', xScale(-value))
        .attr('y', yOffset)
        .attr('text-anchor', 'middle')
        .attr('fill', color)
        .attr('font-size', '11px')
        .attr('font-weight', 'bold')
        .text(`${label}: ${(value * 100).toFixed(2)}%`);
    };

    drawVarLine(var95, 'VaR 95%', '#f97316', -5);
    drawVarLine(var99, 'VaR 99%', '#dc2626', 15);

    // CVaR region
    const cvarBins = bins.filter(d => ((d.x0 as number) + (d.x1 as number)) / 2 <= -var95);
    if (cvarBins.length > 0) {
      g.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', xScale(-var95))
        .attr('height', innerHeight)
        .attr('fill', '#ef4444')
        .attr('opacity', 0.1);

      g.append('text')
        .attr('x', xScale(-cvar95))
        .attr('y', innerHeight - 10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ef4444')
        .attr('font-size', '10px')
        .text(`CVaR: ${(cvar95 * 100).toFixed(2)}%`);
    }

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d => `${(+d * 100).toFixed(1)}%`))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    // Axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 10)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('Daily Returns');

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('Frequency');

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f8fafc')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Return Distribution & Value at Risk');
  }, [returns, var95, var99, cvar95, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default VaRChart;
