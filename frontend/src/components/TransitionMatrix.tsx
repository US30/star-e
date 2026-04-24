import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { scaleSequential } from 'd3-scale';
import { interpolateBlues } from 'd3-scale-chromatic';

interface TransitionMatrixProps {
  matrix: number[][];
  labels?: string[];
  width?: number;
  height?: number;
}

const TransitionMatrix: React.FC<TransitionMatrixProps> = ({
  matrix,
  labels = ['Bear', 'Sideways', 'Bull'],
  width = 350,
  height = 350,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || matrix.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 50, right: 30, bottom: 30, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const cellSize = Math.min(innerWidth, innerHeight) / matrix.length;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const colorScale = scaleSequential(interpolateBlues).domain([0, 1]);

    // Draw cells
    matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        const cell = g.append('g').attr('transform', `translate(${j * cellSize},${i * cellSize})`);

        cell
          .append('rect')
          .attr('width', cellSize - 2)
          .attr('height', cellSize - 2)
          .attr('fill', colorScale(value))
          .attr('stroke', '#1e293b')
          .attr('stroke-width', 2)
          .attr('rx', 4);

        cell
          .append('text')
          .attr('x', cellSize / 2)
          .attr('y', cellSize / 2)
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', value > 0.5 ? '#ffffff' : '#1e293b')
          .attr('font-size', '14px')
          .attr('font-weight', 'bold')
          .text(`${(value * 100).toFixed(0)}%`);
      });
    });

    // Row labels (From)
    labels.forEach((label, i) => {
      g.append('text')
        .attr('x', -10)
        .attr('y', i * cellSize + cellSize / 2)
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', '12px')
        .text(label);
    });

    // Column labels (To)
    labels.forEach((label, i) => {
      g.append('text')
        .attr('x', i * cellSize + cellSize / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .attr('font-size', '12px')
        .text(label);
    });

    // Title
    svg
      .append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f8fafc')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Transition Probabilities');

    // Axis labels
    g.append('text')
      .attr('x', -margin.left / 2)
      .attr('y', innerHeight / 2)
      .attr('transform', `rotate(-90, ${-margin.left / 2}, ${innerHeight / 2})`)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('From');

    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', -30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('To');
  }, [matrix, labels, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default TransitionMatrix;
