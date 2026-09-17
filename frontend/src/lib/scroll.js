/* Shared Lenis instance so any page can smooth-scroll to a result. */
let lenis = null;

export function setLenis(instance) {
  lenis = instance;
}

export function scrollToId(id, offset = -84) {
  const el = document.getElementById(id);
  if (!el) return;
  if (lenis) lenis.scrollTo(el, { offset, duration: 1.1 });
  else el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

export function scrollTop(immediate = false) {
  if (lenis) lenis.scrollTo(0, { immediate });
  else window.scrollTo({ top: 0, behavior: immediate ? 'auto' : 'smooth' });
}
