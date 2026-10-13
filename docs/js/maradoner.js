(function () {
  var top = document.querySelector('.top');
  var toc = document.querySelector('details.toc');
  var links = toc ? [].slice.call(toc.querySelectorAll('a')) : [];
  var heads = links.map(function (a) { return document.getElementById(a.hash.slice(1)); });
  var wide = window.matchMedia('(min-width: 1101px)');

  // The height of the top bar changes when the tabs wrap; anchors and sticky columns follow it.
  function measure() {
    document.documentElement.style.setProperty('--top-h', top.offsetHeight + 'px');
  }
  // The headings of the page are always listed beside the text and are folded when placed above it.
  function fold() {
    if (toc) toc.open = wide.matches;
  }
  function spy() {
    var line = top.offsetHeight + 24, cur = 0;
    heads.forEach(function (h, i) {
      if (h && h.getBoundingClientRect().top <= line) cur = i;
    });
    links.forEach(function (a, i) { a.classList.toggle('active', i === cur); });
  }
  measure(); fold(); spy();
  window.addEventListener('resize', function () { measure(); spy(); });
  window.addEventListener('scroll', spy, { passive: true });
  if (wide.addEventListener) wide.addEventListener('change', fold);
  if (toc) toc.addEventListener('click', function (e) {
    if (e.target.tagName === 'SUMMARY' && wide.matches) e.preventDefault();
    if (e.target.tagName === 'A' && !wide.matches) toc.open = false;
  });

  // Explanations of terms: shown on hover, on keyboard focus and on a tap.
  var tip = document.createElement('div');
  tip.id = 'tip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true;
  document.body.appendChild(tip);
  var shown = null;
  function show(el) {
    shown = el;
    tip.textContent = el.getAttribute('data-tip');
    tip.hidden = false;
    var r = el.getBoundingClientRect(), w = tip.offsetWidth, vw = document.documentElement.clientWidth;
    var x = Math.max(16, Math.min(r.left + r.width / 2 - w / 2, vw - w - 16));
    var y = r.top - tip.offsetHeight - 8;
    if (y < top.offsetHeight + 4) y = r.bottom + 8;
    tip.style.left = x + window.scrollX + 'px';
    tip.style.top = y + window.scrollY + 'px';
  }
  function hide() { shown = null; tip.hidden = true; }
  [].forEach.call(document.querySelectorAll('.tip[data-tip]'), function (el) {
    el.addEventListener('mouseenter', function () { show(el); });
    el.addEventListener('mouseleave', hide);
    el.addEventListener('focus', function () { show(el); });
    el.addEventListener('blur', hide);
    el.addEventListener('click', function (e) { e.stopPropagation(); if (shown !== el) show(el); });
  });
  document.addEventListener('click', hide);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });

  // Diagrams with parts explained on a click: a part of the picture opens the panel named in its data-part.
  [].forEach.call(document.querySelectorAll('.explore'), function (box) {
    var picks = [].slice.call(box.querySelectorAll('.pick'));
    var parts = [].slice.call(box.querySelectorAll('.part'));
    function open(name) {
      picks.forEach(function (p) {
        var on = p.getAttribute('data-part') === name;
        p.classList.toggle('on', on);
        p.setAttribute('aria-expanded', on ? 'true' : 'false');
      });
      parts.forEach(function (s) { s.hidden = s.id !== 'part-' + name; });
    }
    picks.forEach(function (p) {
      function act() { open(p.classList.contains('on') ? '' : p.getAttribute('data-part')); }
      p.addEventListener('click', act);
      p.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); act(); }
      });
    });
    box.classList.add('live');
    open('');
  });

  // Pictures followed step by step: an element lists the steps at which it is shown (data-on), coloured (data-new),
  // crossed out (data-cut), framed (data-goal) or marked as known (data-known). A list item appears at data-from.
  [].forEach.call(document.querySelectorAll('.stepper'), function (box) {
    var range = box.querySelector('input[type=range]');
    var prev = box.querySelector('.prev'), next = box.querySelector('.next'), at = box.querySelector('.at');
    var last = +range.max;
    var marks = ['on', 'new', 'cut', 'goal', 'known'];
    function go(k) {
      k = Math.max(0, Math.min(last, k));
      range.value = k;
      marks.forEach(function (name) {
        [].forEach.call(box.querySelectorAll('[data-' + name + ']'), function (el) {
          el.classList.toggle(name, (' ' + el.getAttribute('data-' + name) + ' ').indexOf(' ' + k + ' ') >= 0);
        });
      });
      [].forEach.call(box.querySelectorAll('[data-from]'), function (el) {
        var from = +el.getAttribute('data-from');
        el.classList.toggle('seen', k >= from);
        el.classList.toggle('now', k >= from && k < +el.getAttribute('data-to'));
      });
      prev.disabled = k === 0;
      next.disabled = k === last;
      at.textContent = k + ' / ' + last;
    }
    range.addEventListener('input', function () { go(+range.value); });
    prev.addEventListener('click', function () { go(+range.value - 1); });
    next.addEventListener('click', function () { go(+range.value + 1); });
    box.classList.add('live');
    go(0);
  });

  if (window.PR) PR.prettyPrint();
})();
