function changeMonth() {
    var selectedYear = document.getElementById('year-select').value;
    var selectedMonth = document.getElementById('month-select').value;

    // URL을 업데이트 하고 페이지를 리로드
    window.location.href = `/pybo/calender?year=${selectedYear}&month=${selectedMonth}`; // URL을 적절히 조정하세요
}