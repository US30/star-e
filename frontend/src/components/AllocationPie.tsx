import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface AllocationPieProps {
  data: { ticker: string; weight: number }[];
  width?: number;
  height?: number;
}

const AllocationPie: React.FC<AllocationPieProps> = ({
  data,
  width = 400,
  height = 400,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const radius = Math.min(width, height) / 2 - 40;
    const innerRadius = radius * 0.6;

    const g = svg
      .append('g')
      .attr('transform', `translate(${width / 2},${height / 2})`);

    const color = d3.scaleOrdinal<string>()
      .domain(data.map(d => d.ticker))
      .range(d3.schemeTableau10);

    const pie = d3.pie<{ ticker: string; weight: number }>()
      .value(d => d.weight)
      .sort(null);

    const arc = d3.arc<d3.PieArcDatum<{ ticker: string; weight: number }>>()
      .innerRadius(innerRadius)
      .outerRadius(radius);

    const outerArc = d3.arc<d3.PieArcDatum<{ ticker: string; weight: number }>>()
      .innerRadius(radius * 1.1)
      .outerRadius(radius * 1.1);

    const arcs = pie(data);

    // Draw slices
    g.selectAll('path')
      .data(arcs)
      .join('path')
      .attr('d', arc)
      .attr('fill', d => color(d.data.ticker))
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 2)
      .on('mouseover', function (event, d) {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('transform', function () {
            const centroid = arc.centroid(d);
            return `translate(${centroid[0] * 0.05},${centroid[1] * 0.05})`;
          })
          .attr('filter', 'brightness(1.2)');
      })
      .on('mouseout', function () {
        d3.select(this)
          .transition()
          .duration(200)
          .attr('transform', 'translate(0,0)')
          .attr('filter', 'none');
      });

    // Labels
    g.selectAll('polyline')
      .data(arcs)
      .join('polyline')
      .attr('stroke', '#64748b')
      .attr('stroke-width', 1)
      .attr('fill', 'none')
      .attr('points', d => {
        const posA = arc.centroid(d);
        const posB = outerArc.centroid(d);
        const posC = outerArc.centroid(d);
        const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1);
        return [posA, posB, posC].map(p => p.join(',')).join(' ');
      })
      .style('opacity', d => (d.data.weight > 0.05 ? 1 : 0));

    g.selectAll('text.label')
      .data(arcs)
      .join('text')
      .attr('class', 'label')
      .attr('transform', d => {
        const pos = outerArc.centroid(d);
        const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        pos[0] = radius * 1.0 * (midangle < Math.PI ? 1 : -1);
        return `translate(${pos})`;
      })
      .attr('text-anchor', d => {
        const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
        return midangle < Math.PI ? 'start' : 'end';
      })
      .attr('fill', '#f8fafc')
      .attr('font-size', '12px')
      .style('opacity', d => (d.data.weight > 0.05 ? 1 : 0))
      .text(d => `${d.data.ticker} (${(d.data.weight * 100).toFixed(1)}%)`);

    // Center text
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#f8fafc')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .text('Portfolio');

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('y', 20)
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('Allocation');
  }, [data, width, height]);

  return <svg ref={svgRef} width={width} height={height} />;
};

export default AllocationPie;
