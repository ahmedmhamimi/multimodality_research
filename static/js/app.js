document.addEventListener('DOMContentLoaded', () => {
  const textareas = document.querySelectorAll('textarea');
  textareas.forEach((el) => {
    el.addEventListener('input', () => {
      el.style.height = 'auto';
      el.style.height = `${Math.max(180, el.scrollHeight)}px`;
    });
  });
});
