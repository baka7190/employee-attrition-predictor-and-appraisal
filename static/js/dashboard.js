window.onload = function () {
  new Chart(document.getElementById("ratingChart"), {
    type: 'bar',
    data: {
      labels: ['1', '2', '3', '4', '5'],
      datasets: [{
        label: 'Count',
        data: [5, 14, 33, 26, 28],
        backgroundColor: '#00cfff'
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true }
      }
    }
  });

  new Chart(document.getElementById("statusChart"), {
    type: 'doughnut',
    data: {
      labels: ['Part Time (16)', 'Full Time'],
      datasets: [{
        data: [16, 84],
        backgroundColor: ['#00cfff', '#007bff']
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } }
    }
  });

  new Chart(document.getElementById("salaryChart"), {
    type: 'doughnut',
    data: {
      labels: ['$45k–58k', '$58k–70k', '$70k–80k', '$35k'],
      datasets: [{
        data: [34, 20, 15, 31],
        backgroundColor: ['#00cfff', '#3ccfff', '#66dfff', '#b4f2ff']
      }]
    },
    options: { plugins: { legend: { position: 'bottom' } } }
  });

  new Chart(document.getElementById("departmentChart"), {
    type: 'bar',
    data: {
      labels: ['Accounting', 'Admin', 'Support', 'Finance', 'HR', 'IT', 'Marketing', 'R&D', 'Sales'],
      datasets: [{
        label: 'Employees',
        data: [12, 16, 2, 6, 4, 18, 12, 16, 15],
        backgroundColor: '#00cfff'
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true } }
    }
  });
}
