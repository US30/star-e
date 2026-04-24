import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface RegimeChartProps {
  data: {
    date: Date;
    price: number;
    regime: number;
  }[];
  width?: number;
  height?: number;
}

const regimeColors = ['#ef4444', '#64748b', '#22c55e']; // Bear, Sideways, Bull
const regimeNames = ['Bear', 'Sideways', 'Bull'];

const RegimeChart: React.FC<RegimeChartProps> = ({
  data,
  width = 800,
  height = 400,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 80, bottom: 40, left: 60 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
      .scaleTime()
      .domain(d3.extent(data, (d) => d.date) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3
      .scaleLinear()
      .domain([
        d3.min(data, (d) => d.price) as number * 0.95,
        d3.max(data, (d) => d.price) as number * 1.05,
      ])
      .range([innerHeight, 0]);

    // Add regime background rectangles
    let currentRegime = data[0].regime;
    let startIdx = 0;

    for (let i = 1; i <= data.length; i++) {
      if (i === data.length || data[i].regime !== currentRegime) {
        g.append('rect')
          .attr('x', xScale(data[startIdx].date))
          .attr('y', 0)
          .attr('width', xScale(data[i - 1].date) - xScale(data[startIdx].date))
          .attr('height', innerHeight)
          .attr('fill', regimeColors[currentRegime])
          .attr('opacity', 0.15);

        if (i < data.length) {
          currentRegime = data[i].regime;
          startIdx = i;
        }
      }
    }

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(
        d3
          .axisBottom(xScale)
          .tickSize(-innerHeight)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .attr('stroke', '#334155')
      .attr('stroke-opacity', 0.5);

    g.append('g')
      .attr('class', 'grid')
      .call(
        d3
          .axisLeft(yScale)
          .tickSize(-innerWidth)
          .tickFormat(() => '')
      )
      .selectAll('line')
      .attr('stroke', '#334155')
      .attr('stroke-opacity', 0.5);

    // Price line
    const line = d3
      .line<{ date: Date; price: number; regime: number }>()
      .x((d) => xScale(d.date))
      .y((d) => yScale(d.price))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(data)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat(d3.timeFormat('%b %Y') as any))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat((d) => `$${d}`))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    // Legend
    const legend = svg
      .append('g')
      .attr('transform', `translate(${width - margin.right + 10},${margin.top})`);

    regimeNames.forEach((name, i) => {
      const legendRow = legend.append('g').attr('transform', `translate(0,${i * 25})`);

      legendRow
        .append('rect')
        .attr('width', 15)
        .attr('height', 15)
        .attr('fill', regimeColors[i])
        .attr('opacity', 0.7);

      legendRow
        .append('text')
        .attr('x', 20)
        .attr('y', 12)
        .attr('fill', '#94a3b8')
        .attr('font-size', '12px')
        .text(name);
    });

    // Tooltip
    const tooltip = d3
      .select('body')
      .append('div')
      .attr('class', 'chart-tooltip')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('pointer-events', 'none');

    const bisect = d3.bisector<{ date: Date; price: number; regime: number }, Date>(
      (d) => d.date
    ).left;

    svg
      .append('rect')
      .attr('width', innerWidth)
      .attr('height', innerHeight)
      .attr('transform', `translate(${margin.left},${margin.top})`)
      .attr('fill', 'transparent')
      .on('mousemove', (event) => {
        const [mx] = d3.pointer(event);
        const x0 = xScale.invert(mx - margin.left);
        const i = bisect(data, x0, 1);
        const d = data[i];

        if (d) {
          tooltip
            .style('visibility', 'visible')
            .style('left', `${event.pageX + 15}px`)
            .style('top', `${event.pageY - 30}px`)
            .html(
              `<div class="text-slate-200">
                <div>${d3.timeFormat('%B %d, %Y')(d.date)}</div>
                <div class="font-bold">$${d.price.toFixed(2)}</div>
                <div style="color: ${regimeColors[d.regime]}">${regimeNames[d.regime]}</div>
              </div>`
            );
        }
      })
      .on('mouseout', () => {
        tooltip.style('visibility', 'hidden');
      });

    return () => {
      tooltip.remove();
    };
  }, [data, width, height]);

  return (
    <svg
      ref={svgRef}
      width={width}
      height={height}
      className="overflow-visible"
    />
  );
};

export default RegimeChart;
