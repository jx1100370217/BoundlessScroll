'use strict';

const states = {
  planned: '待生成', queued: '已排队', running: '生成中', rendered: '已生成',
  reviewed: '已复核', failed: '生成失败', receipt: '等待生成结果', offline: '等待连接 ComfyUI',
};
const byId = (id) => document.getElementById(id);
const select = byId('episode');
const film = byId('film');
const slider = byId('history-slider');
const replay = byId('replay');
const dialog = byId('detail-dialog');
const cache = new Map();
const clock = new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' });
const dateTime = new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23' });
let selected = new URLSearchParams(location.search).get('episode');
let request;
let events = [];
let playTimer = null;
let cursor = null;

function make(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value != null) node.textContent = value;
  return node;
}

function change(key, value, render) {
  const signature = JSON.stringify(value);
  if (cache.get(key) === signature) return;
  cache.set(key, signature);
  render();
}

function timecode(value, precise = false) {
  const scaled = Math.round(value * 100);
  const minutes = Math.floor(scaled / 6000);
  const seconds = String(Math.floor(scaled / 100) % 60).padStart(2, '0');
  const fraction = scaled % 100;
  return `${minutes}:${seconds}${precise && fraction ? '.' + String(fraction).padStart(2, '0') : ''}`;
}

function titleFor(project) {
  if (project.episode && /^ep\d+$/.test(project.id)) return `第${project.id.slice(2)}集 · ${project.title}`;
  return project.title;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const button = byId('theme');
  button.textContent = theme === 'light' ? '☾' : '☀';
  button.setAttribute('aria-label', theme === 'light' ? '切换至深色主题' : '切换至浅色主题');
}
setTheme(localStorage.getItem('scroll-theme') || 'dark');
byId('theme').addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  localStorage.setItem('scroll-theme', theme);
  setTheme(theme);
});

function openDetail(title, body) {
  byId('detail-title').textContent = title;
  byId('detail-body').replaceChildren(body);
  dialog.showModal();
}
dialog.addEventListener('close', () => byId('detail-body').replaceChildren());

function scriptContents(scenes) {
  const body = document.createDocumentFragment();
  for (const scene of scenes) {
    const item = make('div', 'script-scene');
    item.append(make('h3', '', scene.heading));
    for (const beat of scene.beats) {
      const line = make('p', '');
      if (beat.speaker) line.append(make('strong', '', `${beat.speaker}　`));
      line.append(document.createTextNode(beat.text));
      item.append(line);
    }
    body.append(item);
  }
  return body;
}

function drawScript(title, scenes, length, verified) {
  const paper = make('div', 'paper');
  if (verified) paper.append(make('span', 'stamp', '已验收'));
  const header = make('div', 'paper-header');
  const meta = [scenes.length ? `${scenes.length} 个场景` : '剧本'];
  if (length) meta.push(timecode(length, true));
  header.append(make('h2', '', title), make('p', 'meta', meta.join(' · ')));
  paper.append(header);
  if (scenes.length) {
    const excerpt = make('div', 'script-teaser');
    excerpt.append(scriptContents([scenes[0]]));
    const expand = make('button', 'expand', '↗ 展开剧本');
    expand.type = 'button';
    expand.addEventListener('click', () => openDetail(title, scriptContents(scenes)));
    paper.append(excerpt, expand);
  } else {
    paper.append(make('p', '', '剧本尚未导入。'));
  }
  byId('screenplay').replaceChildren(paper);
}

function drawStages(stages) {
  const list = stages.map((stage) => {
    const item = make('li', 'stage');
    item.dataset.state = stage.state;
    const link = make('a');
    link.href = `#${stage.section}`;
    if (stage.state === 'running') link.setAttribute('aria-current', 'step');
    const marker = make('span', 'stage-marker', { done: '✓', running: '●', idle: '·' }[stage.state]);
    marker.setAttribute('aria-hidden', 'true');
    link.append(marker, make('span', 'stage-name', stage.label), make('span', 'stage-detail', stage.detail));
    item.append(link);
    return item;
  });
  byId('stages').replaceChildren(...list);
}

function videoFor(url) {
  const video = make('video');
  video.controls = true;
  video.playsInline = true;
  video.preload = 'metadata';
  video.src = url;
  return video;
}

function drawFilm(shots, fps, current, length) {
  const offset = film.scrollLeft;
  const cards = shots.map((shot) => {
    const card = make('article', 'shot');
    card.dataset.shot = shot.id;
    card.dataset.current = String(Boolean(current && current === (shot.label || shot.id)) || shot.state === 'running');
    const label = make('div', 'shot-label');
    const name = shot.title === shot.id ? shot.id : `${shot.id} · ${shot.title}`;
    label.append(make('h3', '', name), make('span', 'meta', timecode(shot.frames / fps, true)));
    const available = shot.video || shot.image;
    const thumbnail = make(available ? 'button' : 'div', 'shot-image');
    if (shot.image) {
      const image = make('img');
      image.alt = '';
      image.loading = 'lazy';
      image.src = shot.image;
      image.addEventListener('load', () => image.classList.toggle('wide', image.naturalWidth > image.naturalHeight));
      thumbnail.append(image);
    } else {
      thumbnail.classList.add('empty');
      if (!shot.video) thumbnail.textContent = '等待关键帧';
    }
    if (shot.video) thumbnail.append(make('span', 'play-mark', '▶'));
    if (available) {
      thumbnail.type = 'button';
      thumbnail.setAttribute('aria-label', `${shot.video ? '预览镜头' : '查看关键帧'} ${name}`);
      thumbnail.addEventListener('click', () => {
        const body = document.createDocumentFragment();
        if (shot.video) body.append(videoFor(shot.video));
        else {
          const image = make('img');
          image.src = shot.image;
          image.alt = name;
          body.append(image);
        }
        if (shot.text) body.append(make('p', '', shot.text));
        openDetail(name, body);
      });
    }
    const state = make('span', 'shot-state', states[shot.state]);
    state.dataset.state = shot.state;
    card.append(label, thumbnail, state);
    if (shot.text) card.append(make('p', 'shot-text', shot.text));
    return card;
  });
  byId('shot-count').textContent = shots.length ? `${shots.length} 个镜头 · ${timecode(length, true)}` : '';
  film.parentElement.hidden = !shots.length;
  let empty = byId('storyboard-empty');
  if (!shots.length && !empty) {
    empty = make('p', 'empty-state', '分镜计划尚未生成。');
    empty.id = 'storyboard-empty';
    film.parentElement.after(empty);
  }
  if (empty) empty.hidden = Boolean(shots.length);
  film.replaceChildren(...cards);
  film.scrollLeft = offset;
  highlightEvent(false);
}

function drawRender(render, fps, length) {
  const host = byId('render-preview');
  if (!render.video) {
    byId('render-info').textContent = '';
    host.replaceChildren(make('p', 'empty-state', '等待导出视频。'));
    return;
  }
  const info = [render.verified ? '原生 1080p · 已验收' : render.preview ? '拼接预览' : '待验收'];
  if (render.geometry) info.push(render.geometry.join('×'));
  info.push(`${fps} fps`, timecode(length, true));
  byId('render-info').textContent = info.join(' · ');
  const previous = host.querySelector('video');
  const video = previous?.getAttribute('src') === render.video ? previous : videoFor(render.video);
  const card = make('div', 'render-card');
  const links = make('div', 'render-actions');
  links.append(make('p', 'meta', render.name));
  const save = make('a', 'button-link', '下载 MP4');
  save.href = render.video;
  save.download = render.name;
  links.append(save);
  if (render.subtitle) {
    const subtitle = make('a', 'button-link', '下载 SRT');
    subtitle.href = render.subtitle;
    subtitle.download = render.name.replace(/\.mp4$/i, '.srt');
    links.append(subtitle);
  }
  if (video === previous) host.firstElementChild.querySelector('.render-actions').replaceWith(links);
  else {
    card.append(video, links);
    host.replaceChildren(card);
  }
}

function drawRecords(records) {
  const items = records.map((record) => {
    const card = make('article', 'record');
    card.append(make('p', 'meta', record.label), make('h3', '', record.title), make('p', '', record.text));
    return card;
  });
  byId('record-list').replaceChildren(...(items.length ? items : [make('p', 'empty-state', '暂无制作记录。')]));
}

function highlightEvent(scroll) {
  const caption = byId('history-caption');
  const event = events[cursor ?? (events.length - 1)];
  caption.textContent = event
    ? `${clock.format(event.time * 1000)} · ${event.shot ? `${event.shot} ${states[event.state]} · ` : ''}${event.title}`
    : '暂无制作记录';
  slider.value = cursor ?? Math.max(0, events.length - 1);
  for (const card of film.children) {
    const found = cursor !== null && event?.shot === card.dataset.shot;
    card.dataset.highlight = String(found);
    if (found && scroll) film.scrollTo({ left: card.offsetLeft, behavior: 'smooth' });
  }
}

function endPlayback() {
  if (playTimer !== null) clearInterval(playTimer);
  playTimer = null;
  replay.textContent = '▶ 重放本次制作';
}

function drawHistory(nextEvents) {
  events = nextEvents;
  slider.max = Math.max(0, events.length - 1);
  slider.disabled = events.length < 2;
  replay.disabled = events.length < 2;
  if (cursor !== null && cursor >= events.length) cursor = null;
  highlightEvent(false);
}
slider.addEventListener('input', () => {
  endPlayback();
  cursor = Number(slider.value);
  highlightEvent(true);
});
replay.addEventListener('click', () => {
  if (playTimer !== null) {
    endPlayback();
    return;
  }
  cursor = 0;
  highlightEvent(true);
  replay.textContent = 'Ⅱ 暂停回放';
  playTimer = setInterval(() => {
    if (cursor + 1 >= events.length) {
      endPlayback();
      cursor = null;
      highlightEvent(false);
      return;
    }
    cursor += 1;
    highlightEvent(true);
  }, 650);
});

function draw(data) {
  if (!data.projects.length) {
    byId('project-title').textContent = '暂无制作项目';
    byId('status').textContent = '工作台已连接';
    return;
  }
  change('projects', data.projects, () => {
    select.replaceChildren(...data.projects.map((project) => new Option(titleFor(project), project.id)));
    select.disabled = false;
  });
  select.value = data.id;
  const title = titleFor(data.projects.find((project) => project.id === data.id));
  document.title = `${title} · 无垠长卷`;
  byId('project-title').textContent = title;
  const status = byId('status');
  if (status.textContent !== data.status) status.textContent = data.status;
  status.dataset.tone = data.tone;
  const engine = byId('engine');
  engine.textContent = data.online ? 'ComfyUI 已连接' : 'ComfyUI 离线';
  engine.dataset.online = String(data.online);
  const meta = byId('project-meta');
  meta.hidden = !data.shots.length;
  meta.textContent = `${data.shots.length} 个镜头 · ${timecode(data.seconds, true)}`;
  byId('updated').textContent = data.updated ? dateTime.format(data.updated * 1000) : '';
  byId('details').hidden = false;
  change('stages', data.stages, () => drawStages(data.stages));
  change('script', [title, data.scenes, data.seconds, data.render.verified], () =>
    drawScript(title, data.scenes, data.seconds, data.render.verified));
  change('film', [data.shots, data.fps, data.current, data.seconds], () =>
    drawFilm(data.shots, data.fps, data.current, data.seconds));
  change('render', [data.render, data.fps, data.seconds], () => drawRender(data.render, data.fps, data.seconds));
  change('records', data.records, () => drawRecords(data.records));
  change('events', data.events, () => drawHistory(data.events));
}

async function refresh() {
  if (request) return;
  const controller = new AbortController();
  request = controller;
  const params = new URLSearchParams();
  if (selected) params.set('episode', selected);
  try {
    const response = await fetch(`/api/state?${params}`, { cache: 'no-store', signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (request !== controller) return;
    draw(data);
    byId('offline').hidden = true;
  } catch (error) {
    if (error.name !== 'AbortError') byId('offline').hidden = false;
  } finally {
    if (request === controller) request = null;
  }
}

select.addEventListener('change', () => {
  selected = select.value;
  const url = new URL(location.href);
  url.searchParams.set('episode', selected);
  history.replaceState(null, '', url);
  endPlayback();
  cursor = null;
  film.scrollLeft = 0;
  if (request) request.abort();
  request = null;
  refresh();
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refresh();
});
setInterval(() => {
  if (!document.hidden) refresh();
}, 2000);
refresh();
