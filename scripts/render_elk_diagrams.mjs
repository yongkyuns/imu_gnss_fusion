import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import ELK from "elkjs/lib/elk.bundled.js";

const repoRoot = process.cwd();
const specDir = path.join(repoRoot, "docs/assets/elk");
const elk = new ELK();

const palette = {
  text: "#263238",
  subtext: "#607D8B",
  edge: "#6F7C84",
  dashed: "#8FA1AA",
  clusterFill: "#F7FAFC",
  clusterStroke: "#CFD8DC",
  inputFill: "#E8F5E9",
  inputStroke: "#43A047",
  coreFill: "#E1F5FE",
  coreStroke: "#0277BD",
  processFill: "#FFF9C4",
  processStroke: "#FBC02D",
  storageFill: "#FFF3E0",
  storageStroke: "#FFB74D",
  warningFill: "#FCE4EC",
  warningStroke: "#E91E63"
};

const roleColors = {
  input: [palette.inputFill, palette.inputStroke],
  core: [palette.coreFill, palette.coreStroke],
  process: [palette.processFill, palette.processStroke],
  storage: [palette.storageFill, palette.storageStroke],
  warning: [palette.warningFill, palette.warningStroke],
  note: ["#FFFFFF", palette.clusterStroke]
};

const sides = ["NORTH", "EAST", "SOUTH", "WEST"];
const oppositeSide = {
  NORTH: "SOUTH",
  SOUTH: "NORTH",
  EAST: "WEST",
  WEST: "EAST"
};

function defaultSourceSide(direction) {
  return direction === "RIGHT" ? "EAST" : "SOUTH";
}

function defaultTargetSide(direction) {
  return oppositeSide[defaultSourceSide(direction)];
}

function portId(nodeId, side) {
  return `${nodeId}.${side.toLowerCase()}`;
}

function portsFor(node) {
  return sides.map((side, index) => ({
    id: portId(node.id, side),
    width: 2,
    height: 2,
    layoutOptions: {
      "elk.port.side": side,
      "elk.port.index": String(index)
    }
  }));
}

function usesFixedPorts(spec) {
  const profile = spec.profile ?? "structured";
  if (profile === "airy" || profile === "balanced") {
    return (spec.edges ?? []).some((edge) => edge.fromSide || edge.toSide);
  }
  return true;
}

function layoutOptions(spec) {
  const layout = spec.layout ?? {};
  const profile = spec.profile ?? "structured";
  const profileDefaults = {
    airy: {
      edgeRouting: "SPLINES",
      forceModelOrder: "false",
      nodeNode: 82,
      nodeNodeBetweenLayers: 112,
      edgeNode: 34,
      edgeEdge: 26
    },
    structured: {
      edgeRouting: "SPLINES",
      forceModelOrder: "true",
      nodeNode: 76,
      nodeNodeBetweenLayers: 108,
      edgeNode: 38,
      edgeEdge: 28
    },
    dense: {
      edgeRouting: "ORTHOGONAL",
      forceModelOrder: "true",
      nodeNode: 72,
      nodeNodeBetweenLayers: 108,
      edgeNode: 42,
      edgeEdge: 30
    },
    balanced: {
      edgeRouting: "SPLINES",
      forceModelOrder: "true",
      nodeNode: 72,
      nodeNodeBetweenLayers: 84,
      edgeNode: 38,
      edgeEdge: 28
    }
  }[profile] ?? {};
  return {
    "elk.algorithm": "layered",
    "elk.direction": spec.direction ?? "RIGHT",
    "elk.edgeRouting": spec.edgeRouting ?? profileDefaults.edgeRouting ?? "SPLINES",
    "elk.spacing.nodeNode": String(layout.nodeNode ?? profileDefaults.nodeNode ?? 76),
    "elk.layered.spacing.nodeNodeBetweenLayers": String(layout.nodeNodeBetweenLayers ?? profileDefaults.nodeNodeBetweenLayers ?? 108),
    "elk.spacing.edgeNode": String(layout.edgeNode ?? profileDefaults.edgeNode ?? 38),
    "elk.spacing.edgeEdge": String(layout.edgeEdge ?? profileDefaults.edgeEdge ?? 28),
    "elk.spacing.portPort": String(layout.portPort ?? 18),
    "elk.spacing.portsSurrounding": "[top=12,left=16,bottom=12,right=16]",
    "elk.layered.spacing.edgeNodeBetweenLayers": String(layout.edgeNodeBetweenLayers ?? 42),
    "elk.layered.spacing.edgeEdgeBetweenLayers": String(layout.edgeEdgeBetweenLayers ?? 34),
    "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
    "elk.layered.considerModelOrder.portModelOrder": "true",
    "elk.layered.crossingMinimization.forceNodeModelOrder": String(layout.forceModelOrder ?? profileDefaults.forceModelOrder ?? "true"),
    "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
    "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
    "elk.layered.edgeRouting.splines.mode": "CONSERVATIVE",
    ...(layout.partitioning ? { "elk.partitioning.activate": "true" } : {})
  };
}

function nodeForSpec(node, spec) {
  const fixedPorts = usesFixedPorts(spec);
  const elkNode = {
    id: node.id,
    title: node.title,
    subtitle: node.subtitle,
    role: node.role ?? "core",
    shape: node.shape ?? "box",
    group: node.group ?? null,
    width: node.width ?? 230,
    height: node.height ?? 68
  };
  if (fixedPorts) {
    elkNode.ports = portsFor(node);
    elkNode.layoutOptions = {
      "elk.portConstraints": "FIXED_SIDE",
      ...(node.layoutOptions ?? {})
    };
  } else if (node.layoutOptions) {
    elkNode.layoutOptions = node.layoutOptions;
  }
  return elkNode;
}

function edgeForSpec(edge, spec) {
  const direction = spec.direction ?? "RIGHT";
  const fixedPorts = usesFixedPorts(spec);
  const fromSide = edge.fromSide ?? defaultSourceSide(direction);
  const toSide = edge.toSide ?? defaultTargetSide(direction);
  return {
    id: edge.id ?? `${edge.from}-${edge.to}`,
    sources: [fixedPorts ? portId(edge.from, fromSide) : edge.from],
    targets: [fixedPorts ? portId(edge.to, toSide) : edge.to],
    sourceNode: edge.from,
    targetNode: edge.to,
    dashed: Boolean(edge.dashed),
    hidden: Boolean(edge.hidden),
    routing: spec.edgeRouting ?? "SPLINES",
    profile: spec.profile ?? "structured",
    fromSide,
    toSide,
    route: edge.route ?? null
  };
}

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

function textLines(node) {
  if (Array.isArray(node.subtitle)) {
    return [node.title, ...node.subtitle].filter(Boolean);
  }
  return [node.title, node.subtitle].filter(Boolean);
}

function groupBounds(nodes, groupId) {
  const members = nodes.filter((node) => node.group === groupId);
  if (members.length === 0) {
    return null;
  }
  const minX = Math.min(...members.map((node) => node.x));
  const minY = Math.min(...members.map((node) => node.y));
  const maxX = Math.max(...members.map((node) => node.x + node.width));
  const maxY = Math.max(...members.map((node) => node.y + node.height));
  return {
    x: minX - 34,
    y: minY - 62,
    width: maxX - minX + 68,
    height: maxY - minY + 96
  };
}

function shiftLayout(graph, dx, dy) {
  for (const node of graph.children) {
    node.x += dx;
    node.y += dy;
  }
  for (const edge of graph.edges) {
    for (const section of edge.sections ?? []) {
      section.startPoint.x += dx;
      section.startPoint.y += dy;
      section.endPoint.x += dx;
      section.endPoint.y += dy;
      for (const point of section.bendPoints ?? []) {
        point.x += dx;
        point.y += dy;
      }
    }
  }
}

function applyLanes(graph, spec) {
  if ((spec.profile ?? "structured") !== "airy") {
    return;
  }
  const laneIds = [...new Set(spec.nodes.map((node) => node.lane).filter((lane) => lane !== undefined))].sort((a, b) => a - b);
  if (laneIds.length === 0) {
    return;
  }
  const top = Math.min(...graph.children.map((node) => node.y));
  const laneSpacing = spec.layout?.laneSpacing ?? 128;
  const laneY = new Map(laneIds.map((lane, index) => [lane, top + index * laneSpacing]));
  for (const node of graph.children) {
    const nodeSpec = spec.nodes.find((candidate) => candidate.id === node.id);
    if (nodeSpec?.lane !== undefined) {
      node.y = laneY.get(nodeSpec.lane);
    }
  }
}

function applyBalancedGrid(graph, spec) {
  if ((spec.profile ?? "structured") !== "balanced") {
    return;
  }
  const nodeSpecs = new Map(spec.nodes.map((node) => [node.id, node]));
  if (!spec.nodes.some((node) => node.row !== undefined && node.column !== undefined)) {
    return;
  }

  const rows = [...new Set(spec.nodes.map((node) => node.row).filter((row) => row !== undefined))].sort((a, b) => a - b);
  const columns = [...new Set(spec.nodes.map((node) => node.column).filter((column) => column !== undefined))].sort((a, b) => a - b);
  const columnWidths = new Map(columns.map((column) => {
    const width = Math.max(...spec.nodes
      .filter((node) => node.column === column)
      .map((node) => node.width ?? 230));
    return [column, width];
  }));
  const rowHeights = new Map(rows.map((row) => {
    const height = Math.max(...spec.nodes
      .filter((node) => node.row === row)
      .map((node) => node.height ?? 68));
    return [row, height];
  }));

  const columnSpacing = spec.layout?.columnSpacing ?? 76;
  const rowSpacing = spec.layout?.rowSpacing ?? 82;
  const originX = spec.layout?.originX ?? 40;
  const originY = spec.layout?.originY ?? 40;

  const columnX = new Map();
  let cursorX = originX;
  for (const column of columns) {
    columnX.set(column, cursorX);
    cursorX += columnWidths.get(column) + columnSpacing;
  }

  const rowY = new Map();
  let cursorY = originY;
  for (const row of rows) {
    rowY.set(row, cursorY);
    cursorY += rowHeights.get(row) + rowSpacing;
  }

  for (const node of graph.children) {
    const specNode = nodeSpecs.get(node.id);
    if (specNode?.row === undefined || specNode?.column === undefined) {
      continue;
    }
    const slotX = columnX.get(specNode.column);
    const slotY = rowY.get(specNode.row);
    const slotWidth = columnWidths.get(specNode.column);
    const slotHeight = rowHeights.get(specNode.row);
    node.x = slotX + (slotWidth - node.width) / 2;
    node.y = slotY + (slotHeight - node.height) / 2;
  }
}

function straightPath(edge) {
  const section = edge.sections?.[0];
  if (!section) {
    return "";
  }
  const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
  return `M ${points.map((point) => `${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" L ")}`;
}

function roundedPath(edge, radius = 18) {
  const section = edge.sections?.[0];
  if (!section) {
    return "";
  }
  const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
  return roundedPointPath(points, radius);
}

function roundedPointPath(points, radius = 18) {
  if (points.length <= 2) {
    return `M ${points.map((point) => `${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" L ")}`;
  }

  const commands = [`M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`];
  for (let index = 1; index < points.length - 1; index += 1) {
    const prev = points[index - 1];
    const curr = points[index];
    const next = points[index + 1];
    const prevVector = { x: prev.x - curr.x, y: prev.y - curr.y };
    const nextVector = { x: next.x - curr.x, y: next.y - curr.y };
    const prevLength = Math.hypot(prevVector.x, prevVector.y);
    const nextLength = Math.hypot(nextVector.x, nextVector.y);
    if (prevLength < 1 || nextLength < 1) {
      continue;
    }
    const curveRadius = Math.min(radius, prevLength / 2, nextLength / 2);
    const before = {
      x: curr.x + (prevVector.x / prevLength) * curveRadius,
      y: curr.y + (prevVector.y / prevLength) * curveRadius
    };
    const after = {
      x: curr.x + (nextVector.x / nextLength) * curveRadius,
      y: curr.y + (nextVector.y / nextLength) * curveRadius
    };
    commands.push(`L ${before.x.toFixed(1)} ${before.y.toFixed(1)}`);
    commands.push(`Q ${curr.x.toFixed(1)} ${curr.y.toFixed(1)} ${after.x.toFixed(1)} ${after.y.toFixed(1)}`);
  }
  const last = points.at(-1);
  commands.push(`L ${last.x.toFixed(1)} ${last.y.toFixed(1)}`);
  return commands.join(" ");
}

function smoothPath(edge) {
  const section = edge.sections?.[0];
  if (!section) {
    return "";
  }
  const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint];
  if (points.length <= 2) {
    return straightPath(edge);
  }

  const commands = [`M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`];
  for (let index = 0; index < points.length - 1; index += 1) {
    const p0 = points[Math.max(0, index - 1)];
    const p1 = points[index];
    const p2 = points[index + 1];
    const p3 = points[Math.min(points.length - 1, index + 2)];
    const c1 = {
      x: p1.x + (p2.x - p0.x) / 6,
      y: p1.y + (p2.y - p0.y) / 6
    };
    const c2 = {
      x: p2.x - (p3.x - p1.x) / 6,
      y: p2.y - (p3.y - p1.y) / 6
    };
    commands.push(`C ${c1.x.toFixed(1)} ${c1.y.toFixed(1)}, ${c2.x.toFixed(1)} ${c2.y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`);
  }
  return commands.join(" ");
}

function airyPath(edge, nodesById) {
  const source = nodesById.get(edge.sourceNode);
  const target = nodesById.get(edge.targetNode);
  if (!source || !target) {
    return smoothPath(edge);
  }
  const start = {
    x: source.x + source.width,
    y: source.y + source.height / 2
  };
  const end = {
    x: target.x,
    y: target.y + target.height / 2
  };
  const dx = Math.max(70, (end.x - start.x) * 0.45);
  if (edge.route === "below") {
    const channelY = Math.max(source.y + source.height, target.y + target.height) + 42;
    const controlX = start.x + (end.x - start.x) * 0.52;
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} Q ${controlX.toFixed(1)} ${channelY.toFixed(1)} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
  }
  if (edge.route === "above") {
    const channelY = Math.min(source.y, target.y) - 42;
    const controlX = start.x + (end.x - start.x) * 0.52;
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} Q ${controlX.toFixed(1)} ${channelY.toFixed(1)} ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
  }
  if (Math.abs(start.y - end.y) < 3) {
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
  }
  return [
    `M ${start.x.toFixed(1)} ${start.y.toFixed(1)}`,
    `C ${(start.x + dx).toFixed(1)} ${start.y.toFixed(1)}, ${(end.x - dx).toFixed(1)} ${end.y.toFixed(1)}, ${end.x.toFixed(1)} ${end.y.toFixed(1)}`
  ].join(" ");
}

function pointOnSide(node, side) {
  switch (side) {
    case "NORTH":
      return { x: node.x + node.width / 2, y: node.y };
    case "SOUTH":
      return { x: node.x + node.width / 2, y: node.y + node.height };
    case "EAST":
      return { x: node.x + node.width, y: node.y + node.height / 2 };
    case "WEST":
      return { x: node.x, y: node.y + node.height / 2 };
    default:
      return { x: node.x + node.width / 2, y: node.y + node.height };
  }
}

function balancedPath(edge, nodesById) {
  const source = nodesById.get(edge.sourceNode);
  const target = nodesById.get(edge.targetNode);
  if (!source || !target) {
    return smoothPath(edge);
  }

  const sourceCenter = {
    x: source.x + source.width / 2,
    y: source.y + source.height / 2
  };
  const targetCenter = {
    x: target.x + target.width / 2,
    y: target.y + target.height / 2
  };
  const sameRow = Math.abs(sourceCenter.y - targetCenter.y) < Math.max(source.height, target.height) * 0.75;

  if (sameRow) {
    const sourceToLeft = targetCenter.x < sourceCenter.x;
    const start = {
      x: sourceToLeft ? source.x : source.x + source.width,
      y: sourceCenter.y
    };
    const end = {
      x: sourceToLeft ? target.x + target.width : target.x,
      y: targetCenter.y
    };
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
  }

  const targetAbove = targetCenter.y < sourceCenter.y;
  const fromSide = edge.fromSide ?? (targetAbove ? "NORTH" : "SOUTH");
  const toSide = edge.toSide ?? (targetAbove ? "SOUTH" : "NORTH");
  const start = pointOnSide(source, fromSide);
  const end = pointOnSide(target, toSide);
  const fromHorizontal = fromSide === "EAST" || fromSide === "WEST";
  const toHorizontal = toSide === "EAST" || toSide === "WEST";

  if (Math.abs(start.x - end.x) < 3 || Math.abs(start.y - end.y) < 3) {
    return `M ${start.x.toFixed(1)} ${start.y.toFixed(1)} L ${end.x.toFixed(1)} ${end.y.toFixed(1)}`;
  }

  if (fromHorizontal && !toHorizontal) {
    return roundedPointPath([
      start,
      { x: end.x, y: start.y },
      end
    ], 16);
  }

  if (!fromHorizontal && toHorizontal) {
    return roundedPointPath([
      start,
      { x: start.x, y: end.y },
      end
    ], 16);
  }

  if (fromHorizontal && toHorizontal) {
    const routeX = start.x + (end.x - start.x) / 2;
    return roundedPointPath([
      start,
      { x: routeX, y: start.y },
      { x: routeX, y: end.y },
      end
    ], 16);
  }

  const routeY = start.y + (end.y - start.y) / 2;
  return roundedPointPath([
    start,
    { x: start.x, y: routeY },
    { x: end.x, y: routeY },
    end
  ], 16);
}

function pathForEdge(edge, nodesById) {
  if (edge.profile === "airy") {
    return airyPath(edge, nodesById);
  }
  if (edge.profile === "balanced") {
    return balancedPath(edge, nodesById);
  }
  const hasBends = (edge.sections?.[0]?.bendPoints?.length ?? 0) > 0;
  if (!hasBends) {
    return straightPath(edge);
  }
  return edge.routing === "SPLINES" ? smoothPath(edge) : roundedPath(edge);
}

function drawEdge(edge, nodesById) {
  if (edge.hidden) {
    return "";
  }
  const pathData = pathForEdge(edge, nodesById);
  const color = edge.dashed ? palette.dashed : palette.edge;
  const dash = edge.dashed ? "stroke-dasharray=\"8 8\"" : "";
  return `<path d="${pathData}" fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" ${dash} marker-end="url(#arrow)"/>`;
}

function nodeColor(node) {
  return roleColors[node.role] ?? roleColors.core;
}

function drawNode(node) {
  const [fill, stroke] = nodeColor(node);
  const strokeWidth = node.role === "note" ? 1.5 : 2.5;
  const rounded = node.shape === "chip" ? 12 : 10;
  const dash = node.role === "note" ? "stroke-dasharray=\"8 8\"" : "";
  let shapeSvg;

  if (node.shape === "cylinder") {
    shapeSvg = `
      <path d="M ${node.x} ${node.y + 13} C ${node.x} ${node.y - 4}, ${node.x + node.width} ${node.y - 4}, ${node.x + node.width} ${node.y + 13} L ${node.x + node.width} ${node.y + node.height - 13} C ${node.x + node.width} ${node.y + node.height + 4}, ${node.x} ${node.y + node.height + 4}, ${node.x} ${node.y + node.height - 13} Z" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>
      <path d="M ${node.x} ${node.y + 13} C ${node.x} ${node.y + 30}, ${node.x + node.width} ${node.y + 30}, ${node.x + node.width} ${node.y + 13}" fill="none" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
  } else if (node.shape === "diamond") {
    const centerX = node.x + node.width / 2;
    const centerY = node.y + node.height / 2;
    shapeSvg = `<path d="M ${centerX} ${node.y} L ${node.x + node.width} ${centerY} L ${centerX} ${node.y + node.height} L ${node.x} ${centerY} Z" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`;
  } else {
    shapeSvg = `<rect x="${node.x}" y="${node.y}" width="${node.width}" height="${node.height}" rx="${rounded}" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}" ${dash}/>`;
  }

  const lines = textLines(node);
  const titleSize = node.shape === "chip" ? 14 : 17;
  const subtitleSize = node.role === "note" ? 12 : 12;
  const lineHeights = lines.map((_, index) => index === 0 ? titleSize + 4 : subtitleSize + 3);
  const totalHeight = lineHeights.reduce((sum, value) => sum + value, 0);
  let cursorY = node.y + node.height / 2 - totalHeight / 2 + lineHeights[0] - 4;
  const textSvg = lines.map((line, index) => {
    const size = index === 0 ? titleSize : subtitleSize;
    const color = index === 0 ? palette.text : palette.subtext;
    const svg = `<text x="${node.x + node.width / 2}" y="${cursorY.toFixed(1)}" text-anchor="middle" font-size="${size}" fill="${color}">${escapeXml(line)}</text>`;
    cursorY += lineHeights[Math.min(index + 1, lineHeights.length - 1)] ?? 0;
    return svg;
  }).join("\n");

  return `<g>${shapeSvg}\n${textSvg}</g>`;
}

function renderSvg(spec, graph) {
  const groups = spec.groups ?? [];
  let bounds = groups.map((group) => ({ group, bounds: groupBounds(graph.children, group.id) }))
    .filter((entry) => entry.bounds);

  const minX = Math.min(0, ...graph.children.map((node) => node.x), ...bounds.map((entry) => entry.bounds.x));
  const minY = Math.min(0, ...graph.children.map((node) => node.y), ...bounds.map((entry) => entry.bounds.y));
  if (minX < 32 || minY < 32) {
    shiftLayout(graph, 40 - minX, 40 - minY);
  }

  bounds = groups.map((group) => ({ group, bounds: groupBounds(graph.children, group.id) }))
    .filter((entry) => entry.bounds);
  const maxX = Math.max(...graph.children.map((node) => node.x + node.width), ...bounds.map((entry) => entry.bounds.x + entry.bounds.width)) + 40;
  const maxY = Math.max(...graph.children.map((node) => node.y + node.height), ...bounds.map((entry) => entry.bounds.y + entry.bounds.height)) + 40;

  const clusterSvg = bounds.map(({ group, bounds: box }) => `
    <rect x="${box.x}" y="${box.y}" width="${box.width}" height="${box.height}" rx="14" fill="${palette.clusterFill}" stroke="${palette.clusterStroke}" stroke-width="2" stroke-dasharray="8 8"/>
    <text x="${box.x + box.width / 2}" y="${box.y + 28}" text-anchor="middle" font-size="20" fill="${palette.text}">${escapeXml(group.title ?? group.id)}</text>
  `).join("\n");
  const nodesById = new Map(graph.children.map((node) => [node.id, node]));
  const edgeSvg = graph.edges.map((edge) => drawEdge(edge, nodesById)).join("\n");
  const nodeSvg = graph.children.map(drawNode).join("\n");

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.ceil(maxX)}" height="${Math.ceil(maxY)}" viewBox="0 0 ${Math.ceil(maxX)} ${Math.ceil(maxY)}" role="img" aria-label="${escapeXml(spec.title ?? spec.name)}">
  <defs>
    <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 9 4.5 L 0 9 z" fill="${palette.edge}"/>
    </marker>
  </defs>
  <rect width="100%" height="100%" fill="white"/>
  <g font-family="Arial, sans-serif">
    ${clusterSvg}
    ${edgeSvg}
    ${nodeSvg}
  </g>
</svg>`;
}

async function renderSpec(specPath) {
  const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
  const graph = {
    id: spec.name,
    layoutOptions: layoutOptions(spec),
    children: spec.nodes.map((node) => nodeForSpec(node, spec)),
    edges: spec.edges.map((edge) => edgeForSpec(edge, spec))
  };
  const laidOut = await elk.layout(graph);
  applyLanes(laidOut, spec);
  applyBalancedGrid(laidOut, spec);
  const svg = renderSvg(spec, laidOut);
  const output = path.resolve(repoRoot, spec.output);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, svg);
  return output;
}

const requestedSpecs = process.argv.slice(2);
const specs = requestedSpecs.length > 0
  ? requestedSpecs.map((name) => path.resolve(repoRoot, name))
  : fs.readdirSync(specDir)
    .filter((name) => name.endsWith(".json"))
    .map((name) => path.join(specDir, name));

for (const spec of specs) {
  const output = await renderSpec(spec);
  console.log(output);
}
