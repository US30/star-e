import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { interpolateRdYlGn } from 'd3-scale-chromatic';

interface CorrelationHeatmapProps {
  correlationMatrix: number[][];
  labels: string[];
  width?: number;
  height?: number;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  correlationMatrix,
  labels,
  width = 500,
  height = 500,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || correlationMatrix.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 80, right: 30, bottom: 30, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const cellSize = Math.min(innerWidth, innerHeight) / labels.length;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleSequential(interpolateRdYlGn).domain([-1, 1]);

    // Draw cells
    correlationMatrix.forEach((row, i) => {
      row.forEach((value, j) => {
        const cell = g
          .append('g')
          .attr('transform', `translate(${j * cellSize},${i * cellSize})`);

        cell
          .append('rect')
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('fill', colorScale(value))
          .attr('rx', 2);

        if (cellSize > 30) {
          cell
            .append('text')
            .attr('x', cellSize / 2)
            .attr('y', cellSize / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', Math.abs(value) > 0.5 ? '#ffffff' : '#1e293b')
            .attr('font-size', '10px')
            .text(value.toFixed(2));
        }
      });
    });

    // Row labels
    labels.forEach((label, i) => {
      g.append('text')
        .attr('x', -5)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', '11px')
        .text(label);
    });

    // Column labels
    labels.forEach((label, i) => {
      g.append('text')
        .attr('x', i * cellSize + cellSize / 2)
        .attr('y', -5)
        .attr('text-anchor', 'start')
        .attr('dominant-baseline', 'middle')
        .attr('transform', `rotate(-45, ${i * cellSize + cellSize / 2}, -5)`)
        .attr('fill', '#94a3b8')
        .attr('font-size', '11px')
        .text(label);
    });

    // Color legend
    const legendWidth = 200;
    const legendHeight = 15;
    const legendX = (innerWidth - legendWidth) / 2;
    const legendY = innerHeight + 15;

    const legendScale = d3.scaleLinear().domain([-1, 1]).range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale).tickValues([-1, -0.5, 0, 0.5, 1]);

    const defs = svg.append('defs');
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'correlation-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%');

    const gradientStops = [-1, -0.5, 0, 0.5, 1];
    gradientStops.forEach((stop, i) => {
      gradient
        .append('stop')
        .attr('offset', `${(i / (gradientStops.length - 1)) * 100}%`)
        .attr('stop-color', colorScale(stop));
    });

    g.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#correlation-gradient)');

    g.append('g')
      .attr('transform', `translate(${legendX},${legendY + legendHeight})`)
      .call(legendAxis)
      .selectAll('text')
      .attr('fill', '#94a3b8')
      .attr('font-size', '10px');

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f8fafc')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Asset Correlation Matrix');
  }, [correlationMatrix, labels, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default CorrelationHeatmap;
