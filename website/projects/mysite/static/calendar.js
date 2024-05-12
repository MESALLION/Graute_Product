function changeMonth() {
    var selectedYear = document.getElementById('year-select').value;
    var selectedMonth = document.getElementById('month-select').value;

    // URL을 업데이트 하고 페이지를 리로드
    window.location.href = `/pybo/calendar?year=${selectedYear}&month=${selectedMonth}`; // URL을 적절히 조정하세요
}

function toggleYearSelect() {
    var select = document.getElementById('year-select');
    if (select.classList.contains('open')) {
        select.classList.remove('open');
        select.style.overflowY = 'hidden';
        select.style.maxHeight = '40px';  // 최소 높이 유지
    } else {
        select.classList.add('open');
        select.style.overflowY = 'scroll';
        select.style.maxHeight = '200px';
    }
}

document.getElementById('year-select').addEventListener('mousewheel', function(event) {
    this.scrollTop += (event.deltaY * 1); // 스크롤 속도 조절 가능
    event.preventDefault(); // 기본 스크롤 이벤트 방지
}, false);